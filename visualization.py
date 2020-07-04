# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:00:53 2020

@author: xw
"""
import backbones
import torch
import siamfc
import heads
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2



@torch.no_grad()
def plot_filters_single_channel_big(t):
    
    #setting the rows and columns
    nrows = t.shape[0]*t.shape[2]
    ncols = t.shape[1]*t.shape[3]
    
    
    npimg = np.array(t.detach().numpy(), np.float32)
    npimg = npimg.transpose((0, 2, 1, 3))
    npimg = npimg.ravel().reshape(nrows, ncols)
    
    npimg = npimg.T
    
    fig, ax = plt.subplots(figsize=(ncols/10, nrows/200))    
    imgplot = sns.heatmap(npimg, xticklabels=False, yticklabels=False, cmap='gray', ax=ax, cbar=False)

@torch.no_grad()
def plot_filters_single_channel(t):
    
    #kernels depth * number of kernels
    nplots = t.shape[0]*t.shape[1]
    ncols = 12
    
    nrows = 1 + nplots//ncols
    #convert tensor to numpy image
    npimg = np.array(t.detach().numpy(), np.float32)
    
    count = 0
    fig = plt.figure(figsize=(ncols, nrows))
    
    #looping through all the kernels in each channel
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(t[i, j].detach().numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
   
    plt.tight_layout()
    plt.show()
    
if __name__== '__main__':
    net1=siamfc.Net(backbone=backbones.AlexNetV1(),head=heads.SiamFC(out_scale=0.001))
    netPath='C:/Users/xw/Desktop/Siamese-based-object-tracking/nets/siamfc_alexnet_9000_e47.pth'
    net1.load_state_dict(torch.load(netPath))
    net1.eval()
    Weights=[]
    for m in net1.modules():
        if isinstance(m,nn.Conv2d):
            weight=m.weight.permute(0,2,3,1)
            print(weight.shape)
            Weights.append(weight.detach().numpy())
    
    #conv1=cv2.applyColorMap(siamfc.numpy_to_im(Weights[4][10][:,:,20]),cv2.COLORMAP_JET)
    conv1=siamfc.numpy_to_im(Weights[4][10][:,:,20])
    #print(conv1)
    cv2.imshow('a',cv2.resize(conv1,None,fx=50,fy=50))
    cv2.waitKey(1)
    
    