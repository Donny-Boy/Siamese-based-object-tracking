#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:57:05 2020

Data preparation for tranining, first sample two frames from the dataset, then
doing data augmentation. Finally, crop and resize the frames. Scale the frames
so that the bouding area plus context have a fixed size, which is 127*127 according
to the original paper. 

@author: xiangli
"""

import os
from got10k.datasets import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
import numbers
import torch
from torch.utils.data import Dataset

cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 50,
            'batch_size': 8,
            'num_workers': 7,
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}

root_dir = os.path.expanduser('/Users/xiangli/Desktop/Object Tracking/siamfc-pytorch/data/GOT-10k')
seqs=GOT10k(root_dir,subset='train',return_meta=True)


#Define augumentation(transformations) for the raw images

class Compose(object):
    def __init__(self,transforms):
        self.transforms=transforms
    
    def __call__(self,img):
        for t in self.transforms:
            img=t(img)
        return img


class RandomStretch(object):
    def __init__(self,max_stretch=0.5):
        self.max_stretch=max_stretch
    
    def __call__(self,img):
        interp=np.random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4]
            )
        scale=1.0+np.random.uniform(-self.max_stretch,self.max_stretch)
        out_size=(
            round(img.shape[1]*scale),
            round(img.shape[0]*scale)
            )
        return cv2.resize(img,out_size,interpolation=interp)

class CenterCrop(object):
    
    def __init__(self,size):
        if isinstance(size,numbers.Number):
            self.size=(int(size),int(size))
        else:
            self.size=size
    def __call__(self,img):
        h,w=img.shape[:2]
        tw,th=self.size
        i=round((h-th)/2)
        j=round((w-th)/2)
        
        npad=max(0,-i,-j)
        if npad>0:
            avg_color=np.mean(img,axis=(0,1))
            img=cv2.copyMakeBorder(img,npad,npad,npad,npad,cv2.BORDER_CONSTANT,value=avg_color)
            i+=npad
            j+=npad
        
        return img[i:i+th,j:j+tw]

class RandomCrop(object):
    def __init__(self,size):
        if isinstance(size,numbers.Number):
            self.size=(int(size),int(size))
        else:
            self.size=size
    def __call__(self,img):
        h,w = img.shape[:2]
        tw,th=self.size
        i=np.random.randint(0,h-th+1)
        j=np.random.randint(0,w-tw+1)
        return img[i:i+th,j:j+tw]
    
class ToTensor(object):
    def __call__(self,img):
        return torch.from_numpy(img).float().permute((2,0,1))
    
class SiamFCTransforms(object):
    def __init__(self,exemplar_sz=127,instance_sz=255,context=0.5):
        self.exemplar_sz=exemplar_sz
        self.instance_sz=instance_sz
        self.context=context
        
        self.transforms_z=Compose([
            RandomStretch(),
            CenterCrop(instance_sz-8),
            RandomCrop(instance_sz-2*8),
            CenterCrop(exemplar_sz),
            ToTensor()
            ]
            )
        self.transforms_x=Compose([
            RandomStretch(),
            CenterCrop(instance_sz-8),
            RandomCrop(instance_sz-2*8),
            ToTensor()
            ]
            )
    def crop_and_resize(self,img,center,size,out_size,border_type=cv2.BORDER_CONSTANT,
                        border_value=(0,0,0),
                        interp=cv2.INTER_LINEAR):
        #convert box to corners
        size=round(size)
        corners = np.concatenate((
            np.round(center-(size-1)/2),
            np.round(center-(size-1)/2)+size
            ))
        corners=np.round(corners).astype(int)
        # pad image if necessary
        pads = np.concatenate((-corners[:2], corners[2:] - img.shape[:2]))
        npad = max(0, int(pads.max()))
        if npad > 0:
            img = cv2.copyMakeBorder(img, npad, npad, npad, npad,border_type, value=border_value)

        # crop image patch
        corners = (corners + npad).astype(int)
        patch = img[corners[0]:corners[2], corners[1]:corners[3]]

        # resize to out_size
        patch = cv2.resize(patch, (out_size, out_size),
                       interpolation=interp)

        return patch
    
    def _crop(self,img,box,out_size):
        box=np.array([
            box[1]-1+(box[3]-1)/2,
            box[0]-1+(box[3]-1)/2,
            box[3],box[2]
            ])
        center,target_sz=box[:2],box[2:]
        context=self.context*np.sum(target_sz)
        size=np.sqrt(np.prod(target_sz+context))
        size=size*out_size/self.exemplar_sz
        #print(round(size))
        
        avg_color=np.mean(img,axis=(0,1),dtype=float)
        #print(avg_color)
        interp=np.random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4]
            )
        patch=self.crop_and_resize(img,center,size,out_size,border_value=avg_color,interp=interp)
        return patch
    
    def __call__(self,z,x,box_z,box_x):
        z=self._crop(z,box_z,self.instance_sz)
        x=self._crop(x,box_x,self.instance_sz)
        z=self.transforms_z(z)
        x=self.transforms_x(x)
        return z,x
    
class Pair(Dataset):
    def __init__(self,seqs,transforms=None,pairs_per_seq=1):
        super(Pair,self).__init__()
        self.seqs=seqs
        self.transforms=transforms
        self.pairs_per_seq=pairs_per_seq
        self.indices=np.random.permutation(len(seqs))
        self.return_meta=getattr(seqs,'return_meta',False)
        
    def __len__(self):
        return len(self.indices)*self.pairs_per_seq
    
    def __getitem__(self, index):
        index=self.indices[index%len(self.indices)]
        if self.return_meta:
            img_files,anno,meta=self.seqs[index]
            vis_ratios=meta.get('cover',None)
        else:
            img_files,anno=self.seqs[index][:2]
            vis_ratios=None
        
        val_indices=self._filter(cv2.imread(img_files[0],cv2.IMREAD_COLOR),anno,vis_ratios)
        if len(val_indices)<2:
            index=np.random.choice(len(self))
            return self.__getitem__(index)
        
        # sample a frame pair
        rand_z, rand_x = self._sample_pair(val_indices)

        z = cv2.imread(img_files[rand_z], cv2.IMREAD_COLOR)
        x = cv2.imread(img_files[rand_x], cv2.IMREAD_COLOR)
        z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        
        box_z = anno[rand_z]
        box_x = anno[rand_x]

        item = (z, x, box_z, box_x)
        if self.transforms is not None:
            item = self.transforms(*item)
        
        return item
        
    
    def _sample_pair(self, indices):
        n = len(indices)
        assert n > 0

        if n == 1:
            return indices[0], indices[0]
        elif n == 2:
            return indices[0], indices[1]
        else:
            for i in range(100):
                rand_z, rand_x = np.sort(
                    np.random.choice(indices, 2, replace=False))
                if rand_x - rand_z < 100:
                    break
            else:
                rand_z = np.random.choice(indices)
                rand_x = rand_z

            return rand_z, rand_x
    
    def _filter(self, img0, anno, vis_ratios=None):
        size = np.array(img0.shape[1::-1])[np.newaxis, :]
        areas = anno[:, 2] * anno[:, 3]

        # acceptance conditions
        c1 = areas >= 20
        c2 = np.all(anno[:, 2:] >= 20, axis=1)
        c3 = np.all(anno[:, 2:] <= 500, axis=1)
        c4 = np.all((anno[:, 2:] / size) >= 0.01, axis=1)
        c5 = np.all((anno[:, 2:] / size) <= 0.5, axis=1)
        c6 = (anno[:, 2] / np.maximum(1, anno[:, 3])) >= 0.25
        c7 = (anno[:, 2] / np.maximum(1, anno[:, 3])) <= 4
        if vis_ratios is not None:
            c8 = (vis_ratios > max(1, vis_ratios.max() * 0.3))
        else:
            c8 = np.ones_like(c1)
        
        mask = np.logical_and.reduce(
            (c1, c2, c3, c4, c5, c6, c7, c8))
        val_indices = np.where(mask)[0]

        return val_indices
'''z=cv2.imread(seqs[0][0][1])
x=cv2.imread(seqs[0][0][0])
box_z=seqs[0][1][1]
box_x=seqs[0][1][0]
siamFc=SiamFCTransforms()
z,x=siamFc(z,x,box_z,box_x)
#plt.imshow(z[0])
plt.subplot(1,2,1)
plt.imshow(x[0])
plt.subplot(1,2,2)
plt.imshow(z[0])'''
#img=cv2.imread(image1)
#print(img.shape)
#transform=RandomStretch()
#for i in range(4):
#    plt.subplot(2, 2, i + 1) 
#    img=transform(img)
#    plt.imshow(img)
#box=seqs[0][1][0]
#cv2.rectangle(img,(347,443),(347+429,443+272),(0,255,0),5)
#cv2.imshow('image',img)
#cv2.waitKey()==ord()
#cv2.destroyAllWindows()
#cv2.waitKey(1)
















