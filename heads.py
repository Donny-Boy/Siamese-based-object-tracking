#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:05:16 2020

This is the head of Siamese Fc tracker, it cross correlates(convolutes) two input embeddings
and out put a score map

@author: xiangli
"""
import torch.nn as nn
import torch.nn.functional as F

class SiamFC(nn.Module):
    
    def __init__(self,out_scale=0.001):
        super(SiamFC,self).__init__()
        self.out_scale = out_scale
    
    def forward(self,z,x):
        return self.cross_corr(z,x)*self.out_scale
    
    def cross_corr(self,z,x):
        #fast cross correlation
        nz=z.size(0)
        nx,c,h,w=x.size()
        x=x.view(-1,nz*c,h,w)
        out=F.conv2d(x,z,groups=nz)
        out=out.view(nx,-1,out.size(-2),out.size(-1))
        return out
    
'''if __name__=="__main__":
    import torch
    x=torch.randn(8,256,22,22)
    z=torch.randn(8,256,6,6)
    fc=SiamFC()
    output=fc(z,x)
    print(output.size())'''
    
    