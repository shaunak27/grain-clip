"""train DETR-like loss on DINO results"""
import os
import numpy as np
from regex import D, P
import torch
import torch.backends.cudnn as cudnn
import random
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
import torch.cuda.amp as amp 
import argparse
import logging
import sys
from collections import OrderedDict
import clip
import json
from model import losses
from torch.utils.data import DataLoader
import torch.nn.functional as F
sys.path.insert(0, "../")
import utils.tensorboard_utils as TB
from utils.train_utils import AverageMeter, ProgressMeter, save_runtime_checkpoint, set_path,cosine_scheduler,accuracy
from utils import distributed_utils
from model.losses import WordContrastiveLoss
# from model.metric import sim_matrix
from model.tfm import ObjDecoder, Cross_Attention
from model.box_utils import build_matcher, SetCriterion, compute_box_loss, get_matched_indices
import configs.cc_config as config
from dataloaders.clip_prompts import clip_templates
from dataloaders.cub_annotated import CUBDataset, _transform
import torchvision.datasets as datasets
import wandb
from tokenizer import SimpleTokenizer
from PIL import Image, ImageDraw
import time
from box_supervision import get_preprocessed_image
import model.box_ops as box_ops
from dataloaders.conceptual import *
import torch.distributed.nn
os.environ["WANDB_SILENT"] = "true"

def optim_policy(backbone, lr, wd):
    params = []
    no_decay = ['.ln_', '.bn', '.bias', '.logit_scale', '.entropy_scale'] #: modified
    param_group_no_decay = []
    param_group_with_decay = []

    for name, param in backbone.named_parameters():
        if not param.requires_grad:
            print(f'Param not requires_grad: {name}')
            continue
        if any([i in name for i in no_decay]):
            param_group_no_decay.append(param)
        else:
            param_group_with_decay.append(param)

    params.append({'params': param_group_no_decay, 'weight_decay': 0.0})
    params.append({'params': param_group_with_decay, 'weight_decay': wd})

    return params
def make_descriptor_sentence(descriptor):
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f" which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f" which {descriptor}"
    elif descriptor.startswith('used'):
        return f" which is {descriptor}"
    else:
        return f" which has {descriptor}"


def train_and_eval(loader,val_loader, backbone, optimizer, grad_scaler, device, epoch, args, best_acc,lr_schedule,clip_criterion):
    batch_time = AverageMeter('Time',':.2f')
    data_time = AverageMeter('Data',':.4f')
    progress = ProgressMeter(
        len(loader), 
        [batch_time, data_time],
        prefix='Epoch:[{}]'.format(epoch))

    # freeze backbone
    backbone.train()
    end = time.time()
    tic = time.time()
    optimizer.zero_grad()

    for data_idx, data in enumerate(loader):
        
        data_time.update(time.time() - end)

        images,text,obj_boxes,desc,avail_idx = data

        it = loader.num_batches*epoch + data_idx // args.update_freq
        for k, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it]
        if data_idx ==0:
            print('LR is', lr_schedule[it])

        images = images.cuda(device)
        text = text.cuda(device)
        obj_boxes = obj_boxes.cuda(device)
        desc = desc.flatten(0,1)
        avail_idx = avail_idx.flatten(0,1)
        desc = desc[avail_idx==1]
        desc = desc.cuda(device)
        B = images.shape[0]

        loss_dict = {}
        loss_dict['loss'] = 0

        with amp.autocast():
            output = backbone(images,text,desc,return_hidden_state=False,use_desc_train=args.use_desc_train)

            # # ======================== CLIP Loss ================================
            
            loss_dict = clip_criterion(output['image_embed'],output['text_embed'],output['logit_scale'])

        # backward

        grad_scaler.scale(loss_dict['loss']).backward()
        grad_scaler.unscale_(optimizer) #: this is questionable!
        grad_scaler.step(optimizer)
        grad_scaler.update()
        backbone.zero_grad(set_to_none=True)
        backbone.module.logit_scale.data.clamp_(0, 4.6052)
    

        if data_idx == 0:
            avg_meters = {k: AverageMeter(f'{k}:',':.2e') for k in loss_dict.keys()}
        for metric, value in loss_dict.items():
            avg_meters[metric].update(value.item(), B)


        batch_time.update(time.time() - end)
        if data_idx % 10 == 0:
            print(f'backward time {time.time()-end}')
            progress.display(data_idx)
            print('\t' + ' '.join([f"{k}:{v.avg:.2e}" for k,v in avg_meters.items()]))

        end = time.time()
        args.iteration += 1

    progress.synchronize()
    print(f'epoch {epoch} finished, takes {time.time() - tic} seconds')

    return

def test(val_loader, backbone, device,epoch=0):
    backbone.eval()
    #acc = 0
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    class_file = json.load(open(config.imagenet_vondrick_description,'r'))
    aggregated_feats = {}
    descy = {}
    list_of_classes = list(class_file.keys())
    list_of_descriptions = list(class_file.values())
    templates = clip_templates
    for class_idx, classname in enumerate(list_of_classes):
        texty = [t.format(classname) +make_descriptor_sentence(desc) for desc in class_file[classname] for t in templates]
        if False:
            descy[classname] = clip.tokenize([desc for desc in class_file[classname]]).cuda(device)
        aggregated_feats[classname] = clip.tokenize(texty).cuda(device)
        with amp.autocast():
            with torch.no_grad():
                try:
                    aggregated_feats[classname] = backbone.encode_text(aggregated_feats[classname])
                    aggregated_feats[classname] = F.normalize(aggregated_feats[classname],dim=-1)
                    if False:
                        descy[classname] = backbone.encode_text(descy[classname])
                        descy[classname] = F.normalize(descy[classname],dim=-1)
                    aggregated_feats[classname] = aggregated_feats[classname].mean(dim=0)
                except AttributeError:
                    aggregated_feats[classname] = backbone.module.encode_text(aggregated_feats[classname])
                    aggregated_feats[classname] = F.normalize(aggregated_feats[classname],dim=-1)
                    if False:
                        descy[classname] = backbone.module.encode_text(descy[classname])
                        descy[classname] = F.normalize(descy[classname],dim=-1)
                    aggregated_feats[classname] = aggregated_feats[classname].mean(dim=0)
                
    box_loss = 0
    end = time.time()
    for data_idx, data in enumerate(val_loader):
        # ======================== Aggregate Input =====================================
        images,targets = data
        #obj_boxes = obj_boxes.to(device)
        with amp.autocast():
            with torch.no_grad():
                try:
                    img_embeds = backbone.encode_image(images.cuda(device))
                except AttributeError:
                    img_embeds = backbone.module.encode_image(images.cuda(device))
                B = images.shape[0]

                img_embeds = F.normalize(img_embeds,dim=-1)
                
                #measure classification accuracy by computing similarites betwen img_embeds and aggregated_feats, taking the argmax and comparing to the target
                sim_matrix = torch.matmul(img_embeds, torch.stack(list(aggregated_feats.values())).T)
                
                logits_per_image = sim_matrix 
                
                labels = targets.cuda(device)
                acc1, acc5 = accuracy(logits_per_image, labels, topk=(1, 5))
                acc1, acc5 = distributed_utils.scaled_all_reduce([acc1, acc5])
                top1.update(acc1.item(), B)
                top5.update(acc5.item(), B)

                #measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                #progress.display(data_idx)



    progress.synchronize()
    print('0-shot * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    #wandb.log({'val loss': box_loss, 'custom_step': epoch+1})

    return {'acc1': top1.avg, 'acc5': top5.avg}


def setup(args):
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)
    cudnn.benchmark = True
    if distributed_utils.is_main_process():
        prefix = "real_clip_"
        prefix = prefix + (f"{args.num_queries}_queries_")
        args.wandb_name = f"{prefix}_{args.use_desc_train}_{args.backbone.replace('/','_')}_ep{args.epochs}_bs{args.batch_size}_lr{args.lr}"
        args.log_path, args.model_path, args.exp_path = set_path(args,make_dir=args.make_dir)
        
    args.iteration = 1

best_acc = 0

def main(args):
    
    device = distributed_utils.init_distributed_mode(args)
    global best_acc
    
    # seed = args.seed + distributed_utils.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    setup(args)
    print("DEVICE IS", device)
    if distributed_utils.is_main_process():
        wandb.init(project="galav4",name=args.wandb_name)
        wandb.define_metric('custom_step')
        wandb.define_metric('top1',step_metric='custom_step')
        wandb.define_metric('top5',step_metric='custom_step')

    #wandb.define_metric('train loss',step_metric='custom_step')
    #wandb.define_metric('val loss',step_metric='custom_step')
    ### model ###
    #backbone, preprocess = clip.create_model(args.backbone)
    #backbone, preprocess = clip.load('ViT-B/16', device=device, jit=False)
    backbone = getattr(clip.custom_models, args.backbone)()


    tfms = _transform(224)
    
    #val_dataset is Imagenet
    val_dataset = datasets.ImageFolder(os.path.join(args.val_data_dir, 'val'), transform=tfms)

    if len(args.device_ids) > 1:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)
    tokenizer = SimpleTokenizer()
    data = get_data(args,tfms,tokenizer=tokenizer)
    print('Dataset Size:', data['train'].dataloader.num_samples)
    train_loader = data['train'].dataloader

    loader_len = train_loader.num_batches

    args.device_ids = [int(i[0]) for i in args.device_ids]
    backbone.cuda(device)
    
    if len(args.device_ids) > 1:
        
        backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[args.device],bucket_cap_mb=200,find_unused_parameters=True)
    ### optimizer ###
    criterion = losses.CLIPLoss().cuda(args.device)
    params = optim_policy(backbone, args.lr, args.wd)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd,betas=args.betas,eps=args.eps)

    grad_scaler = amp.GradScaler()
    
    if args.resume:
        print(f'loading model from {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu')
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
        args.start_epoch = checkpoint['epoch']+1
        args.iteration = checkpoint['iteration']
        best_acc = checkpoint['best_acc']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Resuming from epoch {args.start_epoch} at iteration {args.iteration}")
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and k != 'step':
                    state[k] = v.cuda()

    lr_schedule = cosine_scheduler(args.lr, args.lr_end, args.epochs,
                                         loader_len // args.update_freq,
                                         warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start)
    
    for epoch in range(args.start_epoch, args.epochs):
        # np.random.seed(args.iteration//2000)
        # random.seed(args.iteration//2000)
        # torch.manual_seed(args.iteration//2000)
        if args.distributed:
            data['train'].set_epoch(epoch)
        train_loader = data['train'].dataloader
        train_and_eval(train_loader, val_loader, backbone, optimizer, grad_scaler, device, epoch, args, best_acc,lr_schedule,criterion)

        if (epoch + 1) % config.eval_every == 0:
            test_acc = test(val_loader, backbone, device,epoch)
            
            is_best = test_acc['acc1'] > best_acc
            best_acc = max(test_acc['acc1'], best_acc)
            if distributed_utils.is_main_process():
                wandb.log({'top1': test_acc['acc1'],'top5': test_acc['acc5'], 'custom_step': epoch+1})
                backbone_state_dict = backbone.state_dict()
                save_dict = {
                    'epoch': epoch,
                    'backbone_state_dict': backbone_state_dict,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'iteration': args.iteration}
                
                prefix = "real_clip_"
                prefix = prefix + (f"{args.num_queries}_queries_")
                save_runtime_checkpoint(save_dict, 
                                        prefix = prefix,
                    filename=os.path.join(args.model_path, 
                    f'ep_{epoch}.pth.tar'),
                    rm_history=False,is_best=is_best)
            # model.cuda(device)
            # backbone.cuda(device)

    if distributed_utils.is_main_process():
        print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))
        wandb.finish()
    
    sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='xattn', type=str)
    parser.add_argument('--seed', default=111,type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument("--loss", nargs="+", default=["a", "b"])
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--is_gala', action='store_true')
    parser.add_argument('--epochs', default=35, type=int)
    parser.add_argument('--alpha',default=1.0,type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--name_prefix', default='', type=str)
    parser.add_argument('--img_path', default='img5', type=str)
    parser.add_argument('--device_ids', default=0, type=list, nargs='+')
    parser.add_argument('--backbone', default='ViT-L/14', type=str)
    parser.add_argument('--results_suffix', default='', type=str)
    parser.add_argument('--meta_dir', default='../data/Clip', type=str)
    parser.add_argument('--data_dir', default='path/to/cc3m/', type=str)
    parser.add_argument('--val_data_dir', default='/path/to/imagenet/', type=str)
    parser.add_argument('--runtime_save_iter', default=2500, type=int)
    parser.add_argument('--optim', default='adamw', type=str)
    parser.add_argument('--num_queries', default=5, type=int)
    parser.add_argument('--loss_bbox_all_boxes',default= 0, type=float)
    parser.add_argument('--loss_giou_all_boxes',default= 0, type=float)
    parser.add_argument('--word_loss_weight', default=0, type=float)
    parser.add_argument('--nce_loss_weight', default=1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--make_dir', default=True, type=bool)
    parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--use_desc_train', action='store_true')
    parser.add_argument('-j', '--num_workers', default=8, type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)


