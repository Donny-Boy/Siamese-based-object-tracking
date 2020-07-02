# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 15:47:29 2020

@author: Xiang Li
"""
import os
import glob
import numpy as np

from siamfc import TrackerSiamFC

if __name__ == '__main__':
    seq_dir=os.path.expanduser('C:/Users/xw/Desktop/Siamese-based-object-tracking/data/OTB/Walking2')
    img_files=sorted(glob.glob(seq_dir+'/img/*.jpg'))
    #anno=np.loadtxt(seq_dir+'/groundtruth_rect.txt',delimiter=',')
    anno=np.loadtxt(seq_dir+'/groundtruth_rect.txt',delimiter='\t')
    net_path='C:/Users/xw/Desktop/Siamese-based-object-tracking/nets/siamfc_alexnetv1_2000_e50.pth'
    #net_path='C:/Users/xw/Desktop/Siamese-based-object-tracking/nets/siamfc_alexnetV2_2000_e50.pth'
    tracker=TrackerSiamFC(net_path=net_path)
    _,_,entropy,peak=tracker.track_with_entropy(img_files,anno[0],visualize='True')
    print('average entropy of the feature maps: ',entropy)
    print('peak value',peak)
    
