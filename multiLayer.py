# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 23:05:15 2020

@author: xw
"""

import torch.nn as nn
import torch
import numpy as np
import siamfc
import os 
from got10k.datasets import *
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import backbones
import heads


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

class AlexNetV1(nn.Module):
    output_stride=8
    
    def __init__(self):
        super(AlexNetV1,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(3,96,11,2),
            nn.BatchNorm2d(96,eps=1e-6,momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2)
            )
        self.conv2=nn.Sequential(
            nn.Conv2d(96,256,5,1,groups=2),
            nn.BatchNorm2d(256,eps=1e-6,momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2)
            )
        self.conv3=nn.Sequential(
            nn.Conv2d(256,384,3,1),
            nn.BatchNorm2d(384,eps=1e-6,momentum=0.05),
            nn.ReLU(inplace=True)
            )
        self.conv4=nn.Sequential(
            nn.Conv2d(384,384,3,1,groups=2),
            nn.BatchNorm2d(384,eps=1e-6,momentum=0.05),
            nn.ReLU(inplace=True)
            )
        self.conv5=nn.Sequential(
            nn.Conv2d(384,256,3,1,groups=2)
            )
        
    def forward(self,x):
        x1=self.conv1(x)
        x2=self.conv2(x1)
        x3=self.conv3(x2)
        x4=self.conv4(x3)
        x5=self.conv5(x4)
        return [x1,x2,x3,x4,x5]

def getScoreMaps(z,x):
    device=torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    length=len(z)
    maps=[]
    for i in range(length):
        a=x[i].to(device)
        b=z[i].to(device)
        out=F.conv2d(a,b)
        #print(out.shape)
        out=torch.squeeze(out)
        out=out.cpu().numpy()
        #print(out.shape)
        maps.append(out)
    return maps

def getMaps(maps):
    new_Maps=[]
    for m in maps:
        m=siamfc.numpy_to_im(m)
        m=cv2.applyColorMap(m,cv2.COLORMAP_JET)
        new_Maps.append(m)
    return new_Maps
        

def cropObject(img, box, exemplar):
    box=np.array([
        box[1]-1+(box[3]-1)/2,
        box[0]-1+(box[2]-1)/2,
        box[3],box[2]],dtype=np.float32
        )
    center,target_sz=box[:2],box[2:]
    #upscale_sz=cfg['response_up']*cfg['response_sz']
    context=cfg['context']*np.sum(target_sz)  
    z_sz=np.sqrt(np.prod(target_sz+context))
    x_sz=z_sz*cfg['instance_sz']/cfg['exemplar_sz']
    avg_color=np.mean(img,axis=(0,1))
    if exemplar==True:
        z=siamfc.crop_and_resize(img, center, z_sz, cfg['exemplar_sz'],border_value=avg_color)
        return z
    elif exemplar==False:
        x=siamfc.crop_and_resize(img, center, x_sz, cfg['instance_sz'],border_value=avg_color)
        return x

def getObject(seqs,task,frame,exemplar):
    img_path=seqs[task][0][frame]
    box=seqs[task][1][frame]
    img=cv2.imread(img_path,cv2.IMREAD_COLOR)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cropped=cropObject(img,box,exemplar)
    #cv2.imshow('a',cropped)
    return cropped

@torch.no_grad()
def getEmbedding(net,img):
    net.eval()
    with torch.no_grad():
        device=torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        net.to(device)
        z=torch.from_numpy(img).to(device).permute(2,0,1).unsqueeze(0).float()
        embeddings=net(z)
    return embeddings
    
    

if __name__=='__main__':
    net=AlexNetV1()
    net_path='C:/Users/xw/Desktop/Siamese-based-object-tracking/nets/siamfc_alexnetv1_2000_e50.pth'
    state_dict=torch.load(net_path)
    state_dict['conv1.0.weight']=state_dict['backbone.conv1.0.weight']
    state_dict['conv1.0.bias']=state_dict['backbone.conv1.0.bias']
    state_dict['conv1.1.weight']=state_dict['backbone.conv1.1.weight']
    state_dict['conv1.1.bias']=state_dict['backbone.conv1.1.bias']
    state_dict['conv1.1.running_mean']=state_dict['backbone.conv1.1.running_mean']
    state_dict['conv1.1.running_var']=state_dict['backbone.conv1.1.running_var']
    state_dict['conv1.1.num_batches_tracked']=state_dict['backbone.conv1.1.num_batches_tracked']
    state_dict['conv2.0.weight']=state_dict['backbone.conv2.0.weight']
    state_dict['conv2.0.bias']=state_dict['backbone.conv2.0.bias']
    state_dict['conv2.1.weight']=state_dict['backbone.conv2.1.weight']
    state_dict['conv2.1.bias']=state_dict['backbone.conv2.1.bias']
    state_dict['conv2.1.running_mean']=state_dict['backbone.conv2.1.running_mean']
    state_dict['conv2.1.running_var']=state_dict['backbone.conv2.1.running_var']
    state_dict['conv2.1.num_batches_tracked']=state_dict['backbone.conv2.1.num_batches_tracked']
    state_dict['conv3.0.weight']=state_dict['backbone.conv3.0.weight']
    state_dict['conv3.0.bias']=state_dict['backbone.conv3.0.bias']
    state_dict['conv3.1.weight']=state_dict['backbone.conv3.1.weight']
    state_dict['conv3.1.bias']=state_dict['backbone.conv3.1.bias']
    state_dict['conv3.1.running_mean']=state_dict['backbone.conv3.1.running_mean']
    state_dict['conv3.1.running_var']=state_dict['backbone.conv3.1.running_var']
    state_dict['conv3.1.num_batches_tracked']=state_dict['backbone.conv3.1.num_batches_tracked']
    state_dict['conv4.0.weight']=state_dict['backbone.conv4.0.weight']
    state_dict['conv4.0.bias']=state_dict['backbone.conv4.0.bias']
    state_dict['conv4.1.weight']=state_dict['backbone.conv4.1.weight']
    state_dict['conv4.1.bias']=state_dict['backbone.conv4.1.bias']
    state_dict['conv4.1.running_mean']=state_dict['backbone.conv4.1.running_mean']
    state_dict['conv4.1.running_var']=state_dict['backbone.conv4.1.running_var']
    state_dict['conv4.1.num_batches_tracked']=state_dict['backbone.conv4.1.num_batches_tracked']
    state_dict['conv5.0.weight']=state_dict['backbone.conv5.0.weight']
    state_dict['conv5.0.bias']=state_dict['backbone.conv5.0.bias']
    net.load_state_dict(state_dict,strict=False)
    net.eval()
    
    root_dir = os.path.expanduser('C:/Users/xw/Desktop/Siamese-based-object-tracking/data/OTB')
    seqs=OTB(root_dir,version=2013)
    x=getObject(seqs,-3,220,False)
    
    #x=torch.from_numpy(x).to(device).permute(2,0,1).unsqueeze(0).float()
    
    z=getObject(seqs,-3,0,True)
    
    #z=torch.from_numpy(z).to(device).permute(2,0,1).unsqueeze(0).float()
    
    #response=net1(z,x)
    #response=torch.squeeze(response.cpu().detach()).numpy()
    #response=siamfc.numpy_to_im(response)
    #plt.imshow(response)
    #plt.imshow(z)
    #plt.imshow(z)
    X=getEmbedding(net,x)
    Z=getEmbedding(net,z)
    maps=getScoreMaps(Z,X)
    m=getMaps(maps)
    for i in range(len(m)):
        m[i]=cv2.resize(m[i], (272, 272),interpolation=cv2.INTER_CUBIC)
    cv2.imshow('a',m[1])
    cv2.waitKey(1)
    #plt.imshow(m[4])
 
    '''state_dict['conv1.0.weight']=state_dict['backbone.conv1.0.weight']
    state_dict['conv1.0.bias']=state_dict['backbone.conv1.0.bias']
    state_dict['conv1.1.weight']=state_dict['backbone.conv1.1.weight']
    state_dict['conv1.1.bias']=state_dict['backbone.conv1.1.bias']
    state_dict['conv1.1.running_mean']=state_dict['backbone.conv1.1.running_mean']
    state_dict['conv1.1.running_var']=state_dict['backbone.conv1.1.running_var']
    state_dict['conv1.1.num_batches_tracked']=state_dict['backbone.conv1.1.num_batches_tracked']
    state_dict['conv2.0.weight']=state_dict['backbone.conv2.0.weight']
    state_dict['conv2.0.bias']=state_dict['backbone.conv2.0.bias']
    state_dict['conv2.1.weight']=state_dict['backbone.conv1.1.weight']
    state_dict['conv2.1.bias']=state_dict['backbone.conv1.1.bias']
    state_dict['conv2.1.running_mean']=state_dict['backbone.conv1.1.running_mean']
    state_dict['conv2.1.running_var']=state_dict['backbone.conv1.1.running_var']
    state_dict['conv2.1.num_batches_tracked']=state_dict['backbone.conv1.1.num_batches_tracked']
    state_dict['conv3.0.weight']=state_dict['backbone.conv2.0.weight']
    state_dict['conv3.0.bias']=state_dict['backbone.conv2.0.bias']
    state_dict['conv3.1.weight']=state_dict['backbone.conv1.1.weight']
    state_dict['conv3.1.bias']=state_dict['backbone.conv1.1.bias']
    state_dict['conv3.1.running_mean']=state_dict['backbone.conv1.1.running_mean']
    state_dict['conv3.1.running_var']=state_dict['backbone.conv1.1.running_var']
    state_dict['conv3.1.num_batches_tracked']=state_dict['backbone.conv1.1.num_batches_tracked']
    state_dict['conv4.0.weight']=state_dict['backbone.conv2.0.weight']
    state_dict['conv4.0.bias']=state_dict['backbone.conv2.0.bias']
    state_dict['conv4.1.weight']=state_dict['backbone.conv1.1.weight']
    state_dict['conv4.1.bias']=state_dict['backbone.conv1.1.bias']
    state_dict['conv4.1.running_mean']=state_dict['backbone.conv1.1.running_mean']
    state_dict['conv4.1.running_var']=state_dict['backbone.conv1.1.running_var']
    state_dict['conv4.1.num_batches_tracked']=state_dict['backbone.conv1.1.num_batches_tracked']
    state_dict['conv5.0.weight']=state_dict['backbone.conv5.0.weight']
    state_dict['conv5.0.bias']=state_dict['backbone.conv5.0.bias']'''
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
