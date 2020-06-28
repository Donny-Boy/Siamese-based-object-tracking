#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 11:18:58 2020

Model training

@author: xiangli
"""
import os
import sys
import torch
import torch.nn as nn
#from .backbones import AlexNetV1
#from .heads import SiamFC
import backbones
import heads
import datasets
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from got10k.datasets import *
import cv2
from got10k.trackers import Tracker
import matplotlib.pyplot as plt
import time
import glob
from torch.utils.tensorboard import SummaryWriter

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


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)

def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels
'''def _create_lables(size):
    def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels
    
    n,c,h,w=size
    x=np.arange(w)-(w-1)/2
    y=np.arange(h)-(h-1)/2
    x,y=np.meshgrid(x,y)
    r_pos = cfg['r_pos'] / cfg['total_stride']
    r_neg = cfg['r_neg'] / cfg['total_stride']
    labels = logistic_labels(x, y, r_pos, r_neg)

    # repeat to size
    labels = labels.reshape((1, 1, h, w))
    labels = np.tile(labels, (n, c, 1, 1))

    # convert to tensors
    labels = torch.from_numpy(labels).float()
        
    return labels'''

class BalancedLoss(nn.Module):
    
    def __init__(self,neg_weight=1.0):
        super(BalancedLoss,self).__init__()
        self.neg_weight=neg_weight
    
    def forward(self,input,target):
        pos_mask = (target == 1)
        neg_mask = (target == 0)
        pos_num = pos_mask.sum().float()
        neg_num = neg_mask.sum().float()
        weight = target.new_zeros(target.size())
        weight[pos_mask] = 1 / pos_num
        weight[neg_mask] = 1 / neg_num * self.neg_weight
        weight /= weight.sum()
        return F.binary_cross_entropy_with_logits(
            input, target, weight, reduction='sum')
    

########################################
'''def train_step(net,batch,optimizer,backward=True):
    net.train(backward)
    z=batch[0]
    x=batch[1]
    
    with torch.set_grad_enabled(backward):
        responses=net(z,x)
        labels=_create_lables(responses.size())
        criterion=BalancedLoss()
        loss=criterion(responses,labels)
        
        if backward:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return loss.item()'''
            
'''def train_over(net,seqs,val_seqs=None,save_dir='pretrained'):
    net.train()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    transforms = datasets.SiamFCTransforms(
            exemplar_sz=cfg['exemplar_sz'],
            instance_sz=cfg['instance_sz'],
            context=cfg['context'])
    
    dataset = datasets.Pair(
            seqs=seqs,
            transforms=transforms)
    
    dataloader = DataLoader(
            dataset,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['num_workers'],
            drop_last=True)
    optimizer=optim.SGD(net.parameters(),lr=cfg['initial_lr'],weight_decay=cfg['weight_decay'],momentum=cfg['momentum'])
    gamma=np.power(cfg['ultimate_lr']/cfg['initial_lr'],1/cfg['epoch_num'])
    lr_scheduler=ExponentialLR(optimizer,gamma)
    for epoch in range(cfg['epoch_num']):
        lr_scheduler.step(epoch=epoch)
        
        for it, batch in enumerate(dataloader):
            loss=train_step(net,batch,optimizer,backward=True)
            print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
            sys.stdout.flush()
        
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
        torch.save(net.state_dict(), net_path)'''

def init_weights(model,gain=1):
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            nn.init.xavier_uniform_(m.weight,gain)
            if m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.BatchNorm2d):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.Linear):
            nn.init.xavier_uniform_(m.weight,gain)
            if m.bias is not None:
                nn.init.constant_(m.bias,0)

def crop_and_resize(img,center,size,out_size,border_type=cv2.BORDER_CONSTANT,
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


def show_image(img, boxes=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    
    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale
    
    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]
        
        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])
        
        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)
        
        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)
    
    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    return img


def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img

def numpy_to_im(a):
    a=a-a.min()
    a=a/a.max()
    a=np.array(a*255,dtype=np.uint8)
    return a 
    

class TrackerSiamFC(Tracker):
    
    def __init__(self,net_path=None,**kwargs):
        super(TrackerSiamFC,self).__init__('SiamFC',True)
        self.cfg=self.parse_args(**kwargs)
        
        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        
        self.net=Net(backbone=backbones.AlexNetV2(),head=heads.SiamFC(out_scale=self.cfg["out_scale"]))
        #init_weights(self.net)
        if net_path is not None:
            self.net.load_state_dict(torch.load(net_path,map_location=lambda storage,loc:storage))
        
        self.net.to(self.device)
        self.criterion=BalancedLoss()
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg['initial_lr'],
            weight_decay=self.cfg['weight_decay'],
            momentum=self.cfg['momentum'])
        # setup lr scheduler
        gamma = np.power(
            self.cfg['ultimate_lr'] / self.cfg['initial_lr'],
            1.0 / self.cfg['epoch_num'])
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)
    
    def parse_args(self,**kwargs):
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
        for key,val in kwargs.items():
            if key in cfg:
                cfg.update({key:val})
        return cfg
    @torch.no_grad()
    def init(self,img,box):
        self.net.eval()
        #convert box to 0-indexed and center based [y,x,h,w]
        box=np.array([
            box[1]-1+(box[3]-1)/2,
            box[0]-1+(box[2]-1)/2,
            box[3],box[2]],dtype=np.float32)
        #print(box)
        self.center,self.target_sz=box[:2],box[2:]
        
        #create hanning window
        self.upscale_sz=self.cfg['response_up']*self.cfg['response_sz']
        self.hann_window=np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window/=self.hann_window.sum()
        self.scale_factors=self.cfg['scale_step']**np.linspace(-(self.cfg['scale_num']//2),self.cfg['scale_num']//2,self.cfg['scale_num'])
        
        #exemplar and search sizes
        context=self.cfg['context']*np.sum(self.target_sz)
        self.z_sz=np.sqrt(np.prod(self.target_sz+context))
        self.x_sz=self.z_sz*self.cfg['instance_sz']/self.cfg['exemplar_sz']
        
        self.avg_color=np.mean(img,axis=(0,1))
        z=crop_and_resize(img,self.center,self.z_sz,out_size=self.cfg['exemplar_sz'],border_value=self.avg_color)
        z=torch.from_numpy(z).to(self.device).permute(2,0,1).unsqueeze(0).float()
        self.kernel=self.net.backbone(z)
        k=torch.Tensor.cpu(self.kernel)
        k=k[0][0].numpy()
        k=numpy_to_im(k)
        k=cv2.applyColorMap(k, cv2.COLORMAP_JET)
        
        cv2.imshow('template',cv2.resize(k,None,fx=15,fy=15))
        cv2.waitKey(1)
      
    @torch.no_grad()
    def update(self,img):
        self.net.eval()
        
        x=[crop_and_resize(img,self.center,self.x_sz*f,out_size=self.cfg['instance_sz'],border_value=self.avg_color) for f in self.scale_factors]
        ##########################
        q=cv2.cvtColor(x[2],cv2.COLOR_BGR2RGB)
        cv2.imshow('search image',q)
        cv2.waitKey(1)
        ##########################
        x=np.stack(x,axis=0)
        x=torch.from_numpy(x).to(self.device).permute(0,3,1,2).float()
        
        #responses
        x=self.net.backbone(x)
        #cv2.imshow('search',cv2.resize(x[0][0].numpy(),None,fx=15,fy=15))
        responses=self.net.head(self.kernel,x)
        responses=responses.squeeze(1).cpu().numpy()
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg['scale_num'] // 2] *= self.cfg['scale_penalty']
        responses[self.cfg['scale_num'] // 2 + 1:] *= self.cfg['scale_penalty']
        
        #peak scale
        scale_id=np.argmax(np.amax(responses,axis=(1,2)))
        #peak location
        response=responses[scale_id]
        response-=response.min()
        #cv2.imshow('peak response',np.log(response))
        #cv2.waitKey(1)
        response/=response.sum()+1e-16
        response=(1-self.cfg['window_influence'])*response+self.cfg['window_influence']*self.hann_window
        peakResponse=numpy_to_im(response)
        peakResponse=cv2.applyColorMap(peakResponse,cv2.COLORMAP_JET)
        cv2.imshow('peakResponse',peakResponse)
        cv2.waitKey(1)
        loc=np.unravel_index(response.argmax(),response.shape)
        
        ###############################
        #show the feature map
        instance=cv2.resize(torch.Tensor.cpu(x[0][0]).numpy(),None,fx=15,fy=15)
        co=np.array(loc)/self.cfg['response_up']*15
        co=(int(co[1]),int(co[0]))
        radius=10
        color=(255,0,0)
        thickness=2
        instance=numpy_to_im(instance)
        instance=cv2.applyColorMap(instance,cv2.COLORMAP_JET)
        instance=cv2.circle(instance,co,radius,color,thickness)
        cv2.imshow('search',instance)
        cv2.waitKey(1)
        ###############################
        
        
        #locate target center
        disp_in_response = np.array(loc)-(self.upscale_sz-1)/2
        disp_in_instance=disp_in_response*self.cfg['total_stride']/self.cfg['response_up']
        disp_in_image=disp_in_instance*self.x_sz*self.scale_factors[scale_id]/self.cfg['instance_sz']
        #print(disp_in_response/self.cfg['response_up'])
        self.center+=disp_in_image
        
        # update target size
        scale =  (1 - self.cfg['scale_lr']) * 1.0 + \
            self.cfg['scale_lr'] * self.scale_factors[scale_id]
        
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])
        return box
    
    
    
    
    def track(self,img_files,box,visualize=False):
        frame_num=len(img_files)
        boxes=np.zeros((frame_num,4))
        boxes[0]=box
        times=np.zeros(frame_num)
        
        for f, img_file in enumerate(img_files):
            img=read_image(img_file)
            begin=time.time()
            if f==0:
                self.init(img,box)
                #print(self.kernel)
            else:
                boxes[f,:]=self.update(img)
            times[f]=time.time()-begin
            
            if visualize:
                show_image(img, boxes[f, :])
        return boxes, times  
   
    def train_step(self,batch,optimizer,backward=True):
        self.net.train(backward)
        z=batch[0].to(self.device, non_blocking=self.cuda)
        x=batch[1].to(self.device, non_blocking=self.cuda)
        with torch.set_grad_enabled(backward):
            responses=self.net(z,x)
            labels=self._create_lables(responses.size())
            loss=self.criterion(responses,labels)
            if backward:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return loss.item()
    @torch.enable_grad()
    def train_over(self,seqs,val_seqs=None,save_dir='pretrained'):
        writer=SummaryWriter()
        val_dir=os.path.expanduser('C:/Users/xw/Desktop/Siamese-based-object-tracking/data/GOT-10k/')
        val_seqs=GOT10k(val_dir, subset='val', return_meta=True)
        self.net.train()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        transforms = datasets.SiamFCTransforms(
            exemplar_sz=self.cfg['exemplar_sz'],
            instance_sz=self.cfg['instance_sz'],
            context=self.cfg['context'])
        
        val_dataset=datasets.Pair(seqs=val_seqs,transforms=transforms)
        val_dataloader=DataLoader(
            val_dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=False,
            num_workers=self.cfg['num_workers'],
            pin_memory=self.cuda,
            drop_last=False
            )
        
        dataset = datasets.Pair(
            seqs=seqs,
            transforms=transforms)
    
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=True,
            num_workers=self.cfg['num_workers'],
            pin_memory=self.cuda,
            drop_last=True)
        for epoch in range(self.cfg['epoch_num']):
            self.lr_scheduler.step(epoch=epoch)
            train_Loss=0
            count1=0
            count2=0
            for it, batch in enumerate(dataloader):
                count1=count1+1
                loss=self.train_step(batch,self.optimizer,backward=True)
                train_Loss=train_Loss+loss
                #count=count+1
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()
            ############################################################
            ######################validation############################
            with torch.no_grad():
                self.net.eval()
                val_loss=0
                for it,batch in enumerate(val_dataloader):
                    z=batch[0].to(self.device, non_blocking=self.cuda)
                    x=batch[1].to(self.device, non_blocking=self.cuda)
                    responses=self.net(z,x)
                    labels=self._create_lables(responses.size())
                    val_loss=val_loss+self.criterion(responses,labels).item()
                    count2=count2+1
                writer.add_scalar('validation loss',val_loss/count2,epoch+1)
                print('average validation loss:',val_loss/count2)
                
            #print(train_Loss)
            writer.add_scalar('training loss',train_Loss/count1,epoch+1)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
        writer.close()     
    def _create_lables(self,size):
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels
    
        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels
    
        n,c,h,w=size
        x=np.arange(w)-(w-1)/2
        y=np.arange(h)-(h-1)/2
        x,y=np.meshgrid(x,y)
        r_pos = self.cfg['r_pos'] / self.cfg['total_stride']
        r_neg = self.cfg['r_neg'] / self.cfg['total_stride']
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        
        return self.labels           
if __name__=="__main__":
    '''root_dir = os.path.expanduser('/Users/xiangli/Desktop/Object Tracking/siamfc-pytorch/data/GOT-10k')
    seqs=GOT10k(root_dir,subset='train',return_meta=True)
    tracker=TrackerSiamFC()
    train_over(net=tracker.net,seqs=seqs,val_seqs=None,save_dir='pretrained')
    '''''''img_files=sorted(glob.glob(seqs_dir+'/*.jpg'))
    box=np.array([181.0000,157.0000,265.0000,302.0000])
    netpath='/Users/xiangli/Desktop/Object Tracking/MySiamFc/siamfc_alexnet_e49.pth'
    tracker=TrackerSiamFC(net_path=netpath)
    tracker.track(img_files,box,visualize=True)'''
    
    
    
    print(torch.cuda.is_available())
    root_dir = os.path.expanduser('C:/Users/xw/Desktop/Siamese-based-object-tracking/data/GOT-10k')
    seqs = GOT10k(root_dir, subset='train', return_meta=True)
    tracker = TrackerSiamFC(response_sz=33,response_up=8,total_stride=4)
    tracker.train_over(seqs)
    
    
    
    
    
    
    
    
    
    
    
            
    
    
    