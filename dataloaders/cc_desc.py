import os
import torch
import numpy as np
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets

class CC_Desc(Dataset):
    def __init__(self, root,n_chunks=1,chunk_idx=0,transform=None):
        self.desc_file = json.load(open(root))
        #divide the dataset into chunks sequentially and choose chunk corresponding to chunk_idx
        
        img_paths = sorted(list(self.desc_file.keys()))
        self.chunk_size = len(img_paths)//n_chunks
        if chunk_idx == n_chunks-1:
            img_paths = img_paths[chunk_idx*self.chunk_size:]
        else:
            img_paths = img_paths[chunk_idx*self.chunk_size:(chunk_idx+1)*self.chunk_size]
        self.imgs = img_paths
        self.transform = transform
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        #caption = self.caps[index].strip()
        img = self.imgs[index]
        desc = self.desc_file[img]
        if self.transform is not None:
            image = self.transform(image)
        #pad desc to 10, if less than 10
        desc = desc[:10]
        for i in range(len(desc)):
            #remove all symbols from start of sentence
            desc[i] = desc[i].lstrip('.,!? -*')
        len_desc = len(desc)
        desc = desc + ['<pad>']*(10-len(desc))
        return image, desc,img,len_desc
