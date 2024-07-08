import logging
import os
import random
from dataclasses import dataclass

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import csv
from tqdm import tqdm
import json
from configs.cc_config import max_boxes
import clip
import random
from dataloaders.cub_annotated import CUBDataset
from dataloaders.clip_prompts import clip_templates

class ConceptualCaptions(Dataset):
    def __init__(self,csv_file,caption_file,bbox_file,data_dir,is_gala=False,transform=None,tokenizer=None):
        self.data_dir = data_dir
        self.transform = transform
        self.is_gala = is_gala
        self.caption_file = json.load(open(caption_file))
        self.bboxes = json.load(open(bbox_file))
        self.captions = []
        self.max_boxes = max_boxes
        self.image_paths = []
        self.tokenizer = tokenizer
        with open(csv_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
               image = row[0]
               caption = row[1]
               if image.endswith(('.jpg','.jpeg','.png')):
                   self.image_paths.append(image)
                   self.captions.append(caption)
        
        self.image_paths = self.image_paths
        self.captions = self.captions

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,idx):
        mypath = self.image_paths[idx]
        if self.transform:
            image = self.transform(image)
        caption = self.captions[idx]
        aug_caption = self.caption_file.get(mypath,caption)
        boxes = torch.zeros((self.max_boxes, 4))
        desc_tensor = torch.zeros((self.max_boxes, 77),dtype=torch.int32)
        avail_idx = torch.zeros(self.max_boxes, dtype=torch.int32)
        if self.bboxes and mypath in self.bboxes:
            
            iboxes = torch.stack([torch.tensor(item['bbox']) for item in self.bboxes.get(mypath,[])])
            iscores = torch.stack([torch.tensor(item['score']) for item in self.bboxes.get(mypath,[])])
            idesc = torch.stack([self.tokenizer(item['description']) for item in self.bboxes.get(mypath,[])])
            topk = iscores.argsort(descending=True)[:self.max_boxes]
            boxes[:len(topk)] = iboxes[topk]
            desc_tensor[:len(topk)] = idesc[topk]
            avail_idx[:len(topk)] = 1
        width = image.shape[1]
        assert width == 224, f'Width is {width}'
        boxes[:, :4] = boxes[:, :4] / 960 * width
        if self.is_gala:
            caption_selection = str(random.choice([caption,aug_caption]))
        else:
            caption_selection = str(caption)
        return image, self.tokenizer(caption_selection), boxes, desc_tensor, avail_idx

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None

    def set_epoch(self, epoch):
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

def get_cc_dataset(args, preprocess_fn, is_train,is_gala=False,tokenizer=None):
    data_dir = args.data_dir
    csv_file = os.path.join(data_dir, 'captions_pooled.csv')
    caption_file = os.path.join(data_dir, 'captions_cc12m_pooled.json')
    bbox_file = os.path.join(data_dir, 'bboxes_cc12m_pooled.json')

    dataset = ConceptualCaptions(csv_file,caption_file,bbox_file,data_dir,is_gala,preprocess_fn,tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=shuffle, 
                            num_workers=args.num_workers, 
                            pin_memory=True, 
                            sampler=sampler,drop_last=is_train)

    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_data(args, preprocess_train,tokenizer):
    if args.is_gala:
        print('Using Gala!!')
    data = {"train": get_cc_dataset(args, preprocess_train, is_train=True,is_gala=args.is_gala,tokenizer=tokenizer)}
    # dataset = CUBDataset(root='#/datasets/CUB/CUB_200_2011/', train=True, transform=preprocess_train, bboxes=True,use_descriptions=True)
    # dataloader = DataLoader(dataset,
    #                         batch_size=args.batch_size,
    #                         shuffle=False,
    #                         num_workers=args.num_workers,
    #                         pin_memory=True,
    #                         drop_last=True,sampler=DistributedSampler(dataset) if args.distributed else None)
    # dataloader.num_samples = len(dataset)
    # dataloader.num_batches = len(dataloader)
    # data = {"train": DataInfo(dataloader)}
    return data