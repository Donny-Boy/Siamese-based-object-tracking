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
   
    
    net_path = '/Users/xiangli/Desktop/Object Tracking/MySiamFc/pretrained/siamfc_alexnet_e50.pth'
    #net_path='/Users/xiangli/Desktop/Object Tracking/MySiamFc/siamfc_alexnet_e49.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    root_dir = os.path.expanduser('/Users/xiangli/Desktop/Object Tracking/siamfc-pytorch/data/OTB')
    e = ExperimentOTB(root_dir, version=2013)
    e.run(tracker,visualize=True)
    e.report([tracker.name])