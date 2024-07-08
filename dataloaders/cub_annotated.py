#Dataloader for CUB-200 dataset annotated with LLM descriptions

import os
import torch
import numpy as np
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from configs.cub_config import annotation_path, bbox_path, max_boxes, max_descriptions,bbox_path_val
try:
    from configs.cub_config import bbox_path_beak, max_boxes_beak
    bbox_path = bbox_path_beak
    max_boxes = max_boxes_beak
except:
    pass
import clip
import random
from dataloaders.clip_prompts import clip_templates
import time
def make_descriptor_sentence(descriptor):
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"
    
class CUBDataset(datasets.ImageFolder):
    """
    Wrapper for the CUB-200-2011 dataset. 
    Method DatasetBirds.__getitem__() returns tuple of image and its corresponding label.    
    Dataset per https://github.com/slipnitskaya/caltech-birds-advanced-classification
    """
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 train=True,
                 use_descriptions=False,
                 split_classes=False,
                 bboxes=False):

        img_root = os.path.join(root, 'images')

        super(CUBDataset, self).__init__(
            root=img_root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )
        
        self.redefine_class_to_idx()

        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train
        self.split_classes = split_classes
        self.max_boxes = max_boxes
        self.use_descriptions = use_descriptions
        # obtain sample ids filtered by split
        path_to_splits = os.path.join(root, 'train_test_split.txt')
        indices_to_use = list()
        with open(path_to_splits, 'r') as in_file:
            for line in in_file:
                idx, use_train = line.strip('\n').split(' ', 2)
                if bool(int(use_train)) == self.train:
                    indices_to_use.append(int(idx))

        # obtain filenames of images
        path_to_index = os.path.join(root, 'images.txt')
        filenames_to_use = set()
        with open(path_to_index, 'r') as in_file:
            for line in in_file:
                idx, fn = line.strip('\n').split(' ', 2)
                if int(idx) in indices_to_use:
                    filenames_to_use.add(fn)

        img_paths_cut = {'/'.join(img_path.rsplit('/', 2)[-2:]): idx for idx, (img_path, lb) in enumerate(self.imgs)}
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]

        _, targets_to_use = list(zip(*imgs_to_use))

        self.imgs = self.samples = imgs_to_use
        #each sample is of the form (path, target)
        #filter out samples with target >= 160 if train is True
        if self.split_classes:
            if self.train:
                self.samples = [sample for sample in self.samples if sample[1] < 160]
            else:
                self.samples = [sample for sample in self.samples if sample[1] >= 160]
    
        self.annotations = json.load(open(annotation_path, 'r'))
        if bboxes:
            # get coordinates of a bounding box
            if self.train:
                self.bboxes = json.load(open(bbox_path, 'r'))
            else:
                self.bboxes = json.load(open(bbox_path_val, 'r'))
        else:
            self.bboxes = None

    def __getitem__(self, index):
        # generate one sample
        #sample, target = super(CUBDataset, self).__getitem__(index)
        #print(len(self.samples))
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.split_classes:
            if self.train:
                assert target < 160
            else:
                assert target >= 160
        #strip path to get category/image name
        path = '/'.join(path.split('/')[-2:])

        if self.transform_ is not None:
            sample = self.transform_(sample)

        if self.bboxes is not None:
            boxes = torch.zeros((self.max_boxes, 4))
            scores = torch.zeros((self.max_boxes))
            desc_tensor = torch.zeros((max_descriptions, 77),dtype=torch.int32)
            avail_idx = torch.zeros(max_descriptions, dtype=torch.int32)
            if max_boxes ==2:
                mypath = path
            else:
                mypath = path.split('/')[1]
            if self.bboxes and mypath in self.bboxes:
                
                iboxes = torch.stack([torch.tensor(item['bbox']) for item in self.bboxes.get(mypath,[])])
                iscores = torch.stack([torch.tensor(item['score']) for item in self.bboxes.get(mypath,[])])
                idesc = torch.stack([clip.tokenize(item['description']).squeeze(0)for item in self.bboxes.get(mypath,[])])
                topk = iscores.argsort(descending=True)[:self.max_boxes]
                boxes[:len(topk)] = iboxes[topk]
                desc_tensor[:len(topk)] = idesc[topk]
                avail_idx[:len(topk)] = 1
            # squeeze coordinates of the bounding box to range [0, 1]
            width = sample.shape[1]
            #filter top 5 boxes by sorting in descending order of confidence stored at item['score']
            
            #box is of the form [x1, y1, x2, y2, score]
            # x1, y1, x2, y2 < 960. Scale it to width
            boxes[:, :4] = boxes[:, :4] / 960 * width
            #print(boxes)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        
        #path is of the form 'category/image_name.jpg'
        descriptions = self.annotations.get(path,[])
        #photo of category
        text = random.choice(clip_templates).format(path.split('/')[0].replace('_', ' ').split('.')[1])
        # desc_tensor = torch.zeros((max_descriptions, 77),dtype=torch.int32)
        
        if len(descriptions) and self.use_descriptions:
            # desc = torch.stack([clip.tokenize(desc).squeeze(0) for desc in descriptions])
            # desc_tensor[:desc.shape[0]] = desc
            text = text + " " + make_descriptor_sentence(random.choice(descriptions))
            # avail_idx[:desc.shape[0]] = 1
        #random.choice(descriptions) if len(descriptions) > 0 else " "
        if self.bboxes is not None:
            return sample, target, clip.tokenize(text,truncate=True).squeeze(0), boxes, desc_tensor, avail_idx
        return sample, target, clip.tokenize(text,truncate=True).squeeze(0),path
    
    def redefine_class_to_idx(self):
        adjusted_dict = {}
        for k, v in self.class_to_idx.items():
            k = k.split('.')[-1].replace('_', ' ')
            split_key = k.split(' ')
            if len(split_key) > 2: 
                k = '-'.join(split_key[:-1]) + " " + split_key[-1]
            adjusted_dict[k] = v
        self.class_to_idx = adjusted_dict



# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
# train_transform = transforms.Compose([
#         transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
#         lambda image: image.convert("RGB"),
#         transforms.ToTensor(),
#         normalize
#     ])

def _transform(n_px):
    #return train_transform
    return transforms.Compose([
        transforms.Resize((n_px,n_px), interpolation=Image.BICUBIC),
        #transforms.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])