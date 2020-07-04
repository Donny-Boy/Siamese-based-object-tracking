#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:04:54 2020

This is the backbone(AlexNet) of the siamese Fc architeture.

@author: xiangli
"""
import torch.nn as nn

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
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        return x

class AlexNetV2(nn.Module):
    output_stride=4
    
    def __init__(self):
        super(AlexNetV2,self).__init__()
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
            nn.MaxPool2d(3,1)
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
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        return x

if __name__=="__main__":
    import torch
    model=AlexNetV2()
    exemplar=torch.randn(1,3,127,127)
    outputEx=model(exemplar)
    searchImage=torch.randn(1,3,255,255)
    outputSeach=model(searchImage)
    
    print('shape 1: ',outputEx.shape)
    print('shape 2: ',outputSeach.shape)
    
    
    
    
    
    
    
    
    
    
    
    
    
    