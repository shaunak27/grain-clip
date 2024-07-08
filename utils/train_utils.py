import torch
from datetime import datetime
import os 
import glob
import json
import sys
import torch
import numpy as np
import pickle
from utils import distributed_utils
import torch.distributed as dist
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from datetime import datetime
from collections import deque
from configs import cc_config as config
def optim_policy(backbone, model, lr, wd):
    params = []
    no_decay = ['.ln_', '.bn', '.bias', '.logit_scale', '.entropy_scale'] #: modified
    param_group_no_decay = []
    param_group_with_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f'Param not requires_grad: {name}')
            continue
        if any([i in name for i in no_decay]):
            param_group_no_decay.append(param)
        else:
            param_group_with_decay.append(param)

    for name, param in backbone.named_parameters():
        if not param.requires_grad:
            print(f'Backbone: Param not requires_grad: {name}')
            continue
        if param.ndim < 2 or 'bias' in name or 'ln' in name or 'bn' in name:
            param_group_no_decay.append(param)
        else:
            param_group_with_decay.append(param)

    params.append({'params': param_group_no_decay, 'weight_decay': 0.0})
    params.append({'params': param_group_with_decay, 'weight_decay': wd})

    return params

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not distributed_utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_runtime_checkpoint(state, prefix,filename, rm_history=True,is_best=False):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    assert filename.endswith('.pth.tar')
    torch.save(state, filename.replace('.pth.tar', f'_{prefix}_{dt_string}.pth.tar'))
    if is_best:
        torch.save(state, filename.replace('.pth.tar', f'_{prefix}_best.pth.tar'))
    print(f'Runtime checkpoint saved to {filename.replace(".pth.tar", f"_{prefix}_{dt_string}.pth.tar")}')
    if rm_history:
        history = sorted(glob.glob(filename.replace('.pth.tar', '_*.pth.tar')))
        if len(history) > 10:
            try:
                history = history[:-10]
                for h in history:os.remove(h)
            except:
                print(f'Caught Error when saving runtime checkpoint: {sys.exc_info()[0]}')
                pass

def set_path(args,make_dir=True):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    args.launch_timestamp = dt_string
    name_prefix = f"{args.name_prefix}" if args.name_prefix else ""
    exp_path = (f"{name_prefix}"
        f"{args.backbone.replace('/','_')}/"
        f"nce_loss{args.nce_loss_weight}_"
        f"word_loss{args.word_loss_weight}_"
        f"bbox_loss{args.loss_bbox_all_boxes}_"
        f"bs{args.batch_size}_lr{args.lr}")
    root = config.ROOT
    log_path = os.path.join(root, exp_path, 'log')
    model_path = os.path.join(root,exp_path, 'model')
    if make_dir and not os.path.exists(log_path): 
        os.makedirs(log_path)
    if make_dir and not os.path.exists(model_path): 
        os.makedirs(model_path)

    with open(f'{log_path}/running_command.txt', 'a') as f:
        json.dump({'command_time_stamp':dt_string, **args.__dict__}, f, indent=2)
        f.write('\n')
    return log_path, model_path, exp_path

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
