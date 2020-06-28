#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:23:49 2020

@author: xiangli
"""
from __future__ import absolute_import
import os
from got10k.experiments import *
from siamfc import TrackerSiamFC

if __name__=='__main__':
   
    
    net_path = 'C:/Users/xw/Desktop/Siamese-based-object-tracking/pretrained/siamfc_alexnet_e50.pth'
    #net_path='C:/Users/xw/Desktop/Siamese-based-object-tracking/nets/siamfc_alexnet_2000_e50.pth'
    
    tracker = TrackerSiamFC(net_path=net_path)

    root_dir = os.path.expanduser('C:/Users/xw/Desktop/Siamese-based-object-tracking/data/OTB')
    e = ExperimentOTB(root_dir, version=2013)
    e.run(tracker,visualize=True)
    e.report([tracker.name])