

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
from utils.train_utils import optim_policy,AverageMeter, ProgressMeter, save_runtime_checkpoint, set_path,cosine_scheduler,accuracy
from utils import distributed_utils
from model.losses import WordContrastiveLoss
# from model.metric import sim_matrix
from model.tfm import ObjDecoder, Cross_Attention
from model.box_utils import build_matcher, SetCriterion, compute_box_loss, get_matched_indices
import configs.cc_config as config
from dataloaders.clip_prompts import clip_templates
from dataloaders.cub_annotated import _transform
from dataloaders.fgvc_data import CUB200Dataset
import torchvision.datasets as datasets
import wandb
from tokenizer import SimpleTokenizer
from PIL import Image, ImageDraw
import time
from box_supervision import get_preprocessed_image
import model.box_ops as box_ops
from dataloaders.conceptual import *
import torch.distributed.nn

def make_descriptor_sentence(descriptor):
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f" which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f" which {descriptor}"
    elif descriptor.startswith('used'):
        return f" which is {descriptor}"
    else:
        return f" which has {descriptor}"

def test(val_loader, model, backbone, device,vondrick_desc_file=None,use_desc=True):
    backbone.eval()
    model.eval()
    #acc = 0
    print("Processing", vondrick_desc_file.split('/')[-1].split('.')[0].split('_')[1])
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    class_file = json.load(open(vondrick_desc_file,'r'))
    aggregated_feats = {}
    descy = {}
    list_of_classes = list(class_file.keys())
    list_of_descriptions = list(class_file.values())
    templates = clip_templates
    for class_idx, classname in enumerate(list_of_classes):
        if use_desc:
            texty = [t.format(classname) + make_descriptor_sentence(desc) for desc in class_file[classname] for t in templates]
        else:
            texty = [t.format(classname) for t in templates]
        descy[classname] = clip.tokenize([desc for desc in class_file[classname]]).cuda(device)
        aggregated_feats[classname] = clip.tokenize(texty).cuda(device)
        with amp.autocast():
            with torch.no_grad():
                try:
                    aggregated_feats[classname] = backbone.encode_text(aggregated_feats[classname]) 
                    aggregated_feats[classname] = F.normalize(aggregated_feats[classname],dim=-1)
                    descy[classname] = backbone.encode_text(descy[classname])
                    descy[classname] = F.normalize(descy[classname],dim=-1)
                    aggregated_feats[classname] = aggregated_feats[classname].mean(dim=0)
                except AttributeError:
                    aggregated_feats[classname] = backbone.module.encode_text(aggregated_feats[classname])
                    aggregated_feats[classname] = F.normalize(aggregated_feats[classname],dim=-1)
                    descy[classname] = backbone.module.encode_text(descy[classname])
                    descy[classname] = F.normalize(descy[classname],dim=-1)
                    aggregated_feats[classname] = aggregated_feats[classname].mean(dim=0)

    end = time.time()
    for data_idx, data in tqdm(enumerate(val_loader),total=len(val_loader)):
        # ======================== Aggregate Input =====================================
        images,targets = data
        #obj_boxes = obj_boxes.to(device)
        with amp.autocast():
            with torch.no_grad():
                try:
                    img_features = backbone.visual(images.cuda(device),return_hidden_state=True)
                except AttributeError:
                    img_features = backbone.module.visual(images.cuda(device),return_hidden_state=True)
                B = images.shape[0]

                model_out, img_embeds, obj_embeds= model(img_features,use_checkpoint=True)

                pred_embed = F.normalize(obj_embeds,dim=-1)
                #take the argmax of the similarity between the pred_embed and the descy
                sim_matrix_desc = {}
                for class_idx, classname in enumerate(list_of_classes):
                    sim_matrix_desc[classname] = torch.matmul(pred_embed, descy[classname].T) # this gives a batch_size * n_queries * n_descriptions tensor
                    #reduce it to a batch_size * n_queries tensor by taking the max over the descriptions
                    sim_matrix_desc[classname] = torch.max(sim_matrix_desc[classname],dim=-1).values
                    #reduce it to a batch_size tensor by taking the mean over the queries
                    sim_matrix_desc[classname] = sim_matrix_desc[classname].mean(dim=-1)
                sim_matrix_desc = torch.stack(list(sim_matrix_desc.values())).T # this gives a batch_size * n_classes tensor
                

                img_embeds = F.normalize(img_embeds,dim=-1)
                
                #measure classification accuracy by computing similarites betwen img_embeds and aggregated_feats, taking the argmax and comparing to the target
                sim_matrix = torch.matmul(img_embeds, torch.stack(list(aggregated_feats.values())).T)
                
                logits_per_image = args.alpha*sim_matrix + (1-args.alpha)*sim_matrix_desc
                
                labels = targets.cuda(device)
                acc1, acc5 = accuracy(logits_per_image, labels, topk=(1, 5))
                acc1, acc5 = distributed_utils.scaled_all_reduce([acc1, acc5])
                top1.update(acc1.item(), B)
                top5.update(acc5.item(), B)

                #measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

    progress.synchronize()
    print(f"{vondrick_desc_file.split('/')[-1].split('.')[0].split('_')[1]:}", '0-shot * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return {'acc1': top1.avg, 'acc5': top5.avg}

def get_data_loader(args,tfms):
    test_loaders = {}
    vondrick_desc_files = {}
    vondrick_root = args.vondrick_root
    if len(args.device_ids) > 1:
        dist = True
    else:
        dist = False
    if 'imagenet' in args.datasets:
        test_dataset = datasets.ImageFolder(os.path.join(args.global_root,'ImageNet/imagenet/' 'val'), transform=tfms)
        if dist:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loaders['imagenet'] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
        vondrick_desc_files['imagenet'] = os.path.join(vondrick_root,'descriptors_imagenet.json')
    if 'imagenet_conflate' in args.datasets:
        test_dataset = datasets.ImageFolder(os.path.join(args.global_root,'ImageNet/imagenet/' 'val'), transform=tfms)
        if dist:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loaders['imagenet'] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
        vondrick_desc_files['imagenet'] = os.path.join(vondrick_root,'descriptors_imagenet_conflate.json')
    if 'cub' in args.datasets:
        test_dataset = CUB200Dataset(os.path.join(args.global_root,'CUB200/CUB_200_2011/'), transform=tfms,train=False)
        if dist:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loaders['cub'] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
        vondrick_desc_files['cub'] = os.path.join(vondrick_root,'descriptors_cub.json')
    if 'imagenetv2' in args.datasets:
        test_dataset = datasets.ImageFolder(os.path.join(args.global_root,'ImageNet/imagenetv2/imagenetv2-matched-frequency-format-val'), transform=tfms)
        if dist:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loaders['imagenetv2'] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
        vondrick_desc_files['imagenetv2'] = os.path.join(vondrick_root,'descriptors_imagenet.json')
    if 'eurosat' in args.datasets:
        test_dataset = datasets.EuroSAT(os.path.join(args.global_root,'eurosat'), transform=tfms,download=False)
        if dist:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loaders['eurosat'] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
        vondrick_desc_files['eurosat'] = os.path.join(vondrick_root,'descriptors_eurosat.json')
    if 'places365' in args.datasets:
        test_dataset = datasets.Places365(os.path.join(args.global_root,'places-365'), transform=tfms,download=False,split='val')
        if dist:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loaders['places365'] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
        vondrick_desc_files['places365'] = os.path.join(vondrick_root,'descriptors_places365.json')
    if 'food101' in args.datasets:
        test_dataset = datasets.Food101(args.global_root, transform=tfms,download=False,split='test')
        if dist:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loaders['food101'] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
        vondrick_desc_files['food101'] = os.path.join(vondrick_root,'descriptors_food101.json')
    if 'oxfordiiitpets' in args.datasets:
        test_dataset = datasets.OxfordIIITPet(args.global_root, transform=tfms,split='test',download=False)
        if dist:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loaders['oxfordiiitpets'] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
        vondrick_desc_files['oxfordiiitpets'] = os.path.join(vondrick_root,'descriptors_pets.json')
    if 'dtd' in args.datasets:
        test_dataset = datasets.DTD(args.global_root, transform=tfms,download=False,split='test')
        if dist:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loaders['dtd'] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
        vondrick_desc_files['dtd'] = os.path.join(vondrick_root,'descriptors_dtd.json')
    if 'cifar10' in args.datasets:
        test_dataset = datasets.CIFAR10(args.global_root, transform=tfms,download=False,train=False)
        if dist:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loaders['cifar10'] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
        vondrick_desc_files['cifar10'] = os.path.join(vondrick_root,'descriptors_cifar10.json')
    if 'cifar100' in args.datasets:
        test_dataset = datasets.CIFAR100(args.global_root, transform=tfms,download=False,train=False)
        if dist:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loaders['cifar100'] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
        vondrick_desc_files['cifar100'] = os.path.join(vondrick_root,'descriptors_cifar100.json')
    if 'sun397' in args.datasets:
        test_dataset = datasets.SUN397(args.global_root, transform=tfms,download=False)
        if dist:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loaders['sun397'] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
        vondrick_desc_files['sun397'] = os.path.join(vondrick_root,'descriptors_sun.json')
    if 'stanfordcars' in args.datasets:
        test_dataset = datasets.ImageFolder(os.path.join(args.global_root,'stanford_cars/cars_test'), transform=tfms)
        if dist:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loaders['stanfordcars'] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
        vondrick_desc_files['stanfordcars'] = os.path.join(vondrick_root,'descriptors_cars.json')
    if 'caltech101' in args.datasets:
        test_dataset = datasets.Caltech101(args.global_root, transform=tfms,download=False)
        if dist:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loaders['caltech101'] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
        vondrick_desc_files['caltech101'] = os.path.join(vondrick_root,'descriptors_caltech.json')
    if 'flowers102' in args.datasets:
        test_dataset = datasets.Flowers102(args.global_root, transform=tfms,download=True,split='test')
        if dist:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loaders['flowers102'] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
        vondrick_desc_files['flowers102'] = os.path.join(vondrick_root,'descriptors_flowers.json')
    
    return test_loaders, vondrick_desc_files

def main(args):
    
    device = distributed_utils.init_distributed_mode(args)
    
    cudnn.benchmark = True
    print("DEVICE IS", device)

    backbone = getattr(clip.custom_models, args.backbone)()
    feature_dim = backbone.visual.embed_dim
    
    
    num_queries = args.num_queries + 1

    tfm = Cross_Attention(normalize_before=True,
                    return_intermediate_dec=True,dropout=0.1)
    model = ObjDecoder(transformer=tfm, 
                    num_classes=22047, # not used
                    num_queries=num_queries, 
                    aux_loss=True,
                    pred_traj=True,
                    feature_dim=feature_dim,
                    self_attn=False)
        
    tfms = _transform(224)
    
    args.datasets = ['imagenet','cub','imagenetv2','places365','food101','oxfordiiitpets','dtd','cifar10','cifar100','sun397','stanfordcars','caltech101','flowers102']

    test_datasets,vondrick_paths = get_data_loader(args,tfms)

    args.device_ids = [int(i[0]) for i in args.device_ids]

    model.cuda(device) 
    backbone.cuda(device)
    
    if len(args.device_ids) > 1:
        
        backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[args.device],bucket_cap_mb=200,find_unused_parameters=True)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device],bucket_cap_mb=200,find_unused_parameters=True)

    
    if args.resume:
        print(f'loading model from {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
        args.start_epoch = checkpoint['epoch']+1
        args.iteration = checkpoint['iteration']
        print(f"Last trained on epoch {args.start_epoch} at iteration {args.iteration}")
    acc_dump = {}
    for dataset in test_datasets:
        val_loader = test_datasets[dataset]
        vondrick_desc_file = vondrick_paths[dataset]     
        test_acc = test(val_loader, model, backbone, device,vondrick_desc_file,args.use_desc)
        acc_dump[dataset] = {'acc1': test_acc['acc1'], 'acc5': test_acc['acc5']}

    if distributed_utils.is_main_process():
        print('Testing complete')
        json.dump(acc_dump,open('zero_shot_results_oureval_gala_cc12m.json','w'))



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--img_path', default='img5', type=str)
    parser.add_argument('--alpha',default = 1,type=float)
    parser.add_argument('--device_ids', default=0, type=list, nargs='+')
    parser.add_argument('--backbone', default='ViT-L/14', type=str)
    parser.add_argument('--global_root', default='/#/datasets/', type=str)
    parser.add_argument('--vondrick_root', default='#/classify_by_description_release/descriptors/descriptors_new/', type=str)
    parser.add_argument('--num_queries', default=5, type=int)
    parser.add_argument('--use_desc', default=False, action='store_true')
    parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('-j', '--num_workers', default=8, type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
