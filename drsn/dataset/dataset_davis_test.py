#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
#
# Distributed under terms of the MIT license.

from torch.utils import data
import os
import sys
import numpy as np
from PIL import Image
import cv2
import math
import json
import random

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURR_DIR)
from davis_api import DAVIS
from datautils import *
from data_augmentation import Augmentor

davis = DAVIS(split='val')

""" For finetune """
class DAVISDataSet_Ft(data.Dataset):
    def __init__(self, img_size=(256, 256), vdname=None, label_id=None):
        self.img_size = img_size
        self.mean_value = np.array((104, 117, 123), np.float)
        self.data_len = 0
        self.auger = Augmentor(rand_rotate_angle=True, data_aug_flip=True, color_aug=True,
                               gaussian_blur=False, motion_blur=False)

    def reset(self, vdname, label_id):
        self.vdname = vdname
        self.label_id = label_id
        self.imglist = davis.get_imglist(vdname)
        self.annolist = davis.get_annolist(vdname)
        self.init_mask = readobjmask(self.annolist[0], int(label_id))
        self.init_image = readimage(self.imglist[0])
        self.ref_gd_box = np.array(get_mask_bbox(self.init_mask))

    def __getitem__(self, index):
        ref_box = expand_box(self.ref_gd_box, self.init_mask.shape, ratio=1.5)
        ref_image, ref_mask = cropimage(self.init_image, ref_box), cropimage(self.init_mask, ref_box)

        def gen_fake_box(box, mask):
            fake_box = expand_box(box, mask.shape, ratio=1.5 + abs(np.random.random()-0.5))
            return shaking_bbox(mask, fake_box) 

        track_box = gen_fake_box(self.ref_gd_box, self.init_mask)
        track_image, track_mask = cropimage(self.init_image, track_box), cropimage(self.init_mask, track_box)

        cand0_box, cand1_box = gen_fake_box(self.ref_gd_box, self.init_mask), gen_fake_box(self.ref_gd_box, self.init_mask)
        cand0_image, cand0_mask = cropimage(self.init_image, cand0_box), cropimage(self.init_mask, cand0_box)
        cand1_image, cand1_mask = cropimage(self.init_image, cand1_box), cropimage(self.init_mask, cand1_box)
        # Resize to proper size
        ref_image, ref_mask, track_image, track_mask = [resize(x, self.img_size) for x in [ref_image, ref_mask, track_image, track_mask]]
        cand0_image, cand0_mask, cand1_image, cand1_mask = [resize(x, self.img_size) for x in [cand0_image, cand0_mask, cand1_image, cand1_mask]]
        prev_gmask = np.array(get_gb_image(track_mask), np.float32)
        
        # Augmentation
        track_image, track_mask = self.auger(track_image, track_mask)
        cand0_image, cand0_mask = self.auger(cand0_image, cand0_mask)
        cand1_image, cand1_mask = self.auger(cand1_image, cand1_mask)

        # ToTensor
        ref_image, track_image = np.array(ref_image, dtype=np.float32), np.array(track_image, dtype=np.float32)  
        cand0_image, cand1_image = np.array(cand0_image, dtype=np.float32), np.array(cand1_image, dtype=np.float32)  
        ref_mask, prev_gmask, track_mask = np.array(ref_mask, dtype=np.uint8), np.array(prev_gmask, dtype=np.float32), np.array(track_mask, dtype=np.uint8)
        cand0_mask, cand1_mask = np.array(cand0_mask, dtype=np.uint8), np.array(cand1_mask, dtype=np.uint8)
        
        DEBUG=False
        if DEBUG:
            cv2.imwrite('ref_image.jpg', ref_image[..., ::-1])
            cv2.imwrite('ref_mask.jpg', ref_mask*255)
            cv2.imwrite('prev_gmask.jpg', prev_gmask*255)
            cv2.imwrite('track_image.jpg', track_image[..., ::-1])
            cv2.imwrite('track_mask.jpg', track_mask*255)
            cv2.imwrite('cand0_mask.jpg', cand0_mask*255)
            cv2.imwrite('cand0_image.jpg', cand0_image[..., ::-1])
            cv2.imwrite('cand1_mask.jpg', cand1_mask*255)
            cv2.imwrite('cand1_image.jpg', cand1_image[..., ::-1])

        # Normalize
        ref_image -= self.mean_value
        track_image -= self.mean_value
        cand0_image -= self.mean_value
        cand1_image -= self.mean_value
        ref_image, track_image = hwc2chw(ref_image), hwc2chw(track_image)
        cand0_image, cand1_image = hwc2chw(cand0_image), hwc2chw(cand1_image)
        prev_gmask = np.expand_dims(prev_gmask, axis=0)
        ref_mask = np.expand_dims(ref_mask, axis=0)
        cand0_mask, cand1_mask = np.expand_dims(cand0_mask, axis=0), np.expand_dims(cand1_mask, axis=0)
        return ref_image.copy(), ref_mask.copy(), \
               cand0_image.copy(), cand0_mask.copy(), \
               cand1_image.copy(), cand1_mask.copy(), \
               prev_gmask.copy(), track_image.copy(), track_mask.copy()

    def __len__(self):
        return self.data_len


""" For inference. """
class DAVISDataSet_Test(data.Dataset):
    def __init__(self, img_size=(256, 256), vdname=None, label_id=None):
        self.img_size = img_size
        self.mean_value = np.array((104, 117, 123), np.float)
        self.vdname = None
        self.label_id = None

    def reset(self, vdname, label_id):
        self.vdname = vdname
        self.label_id = label_id
        self.imglist = davis.get_imglist(vdname)
        self.annolist = davis.get_annolist(vdname)
        self.init_mask = readobjmask(self.annolist[0], int(label_id))
        self.init_image = readimage(self.imglist[0])
        ref_gd_box = get_mask_bbox(self.init_mask)
        ref_box = expand_box(ref_gd_box, self.init_mask.shape, ratio=1.5)
        self.ref_image, self.ref_mask = cropimage(self.init_image, ref_box), cropimage(self.init_mask, ref_box)

        # State variables
        self.prev_mask = self.init_mask
        self.pred_box = np.array(get_mask_bbox(self.init_mask))
        self.origin_size = self.init_mask.shape
        try:
            #MDNET_HOME = os.path.join(CURR_DIR, "mdnet_preds")
            #mdnet_preds = np.load(os.path.join(MDNET_HOME, vdname, str(label_id), 'result.npy'))[:, :5, 1:].mean(axis=1)
            #self.mdnet_preds = remove_mdnet_missing_box(mdnet_preds)
            MDNET_HOME = os.path.join(CURR_DIR, "result_davis")
            with open(os.path.join(MDNET_HOME, vdname, str(label_id), 'result.json'), "r") as f:
                self.mdnet_preds = np.array(json.load(f)['res'])
        except:
            print ('MDNet preds nonexist occured at {} -- {}'.format(vdname, label_id))
            self.mdnet_preds = np.array([cross2otb(self.pred_box)] * len(self.imglist))

    def __getitem__(self, index):
        track_image = readimage(self.imglist[1:][index])
        track_frame_idx = get_img_str_idx(self.imglist[1:][index])
        pred_box = otb2cross(np.array(self.mdnet_preds[index]))
        if (pred_box[3]-pred_box[1])*(pred_box[2]-pred_box[0]) >= self.origin_size[0]*self.origin_size[1] * 0.6:
            pred_box = np.array([0, 0, self.origin_size[1], self.origin_size[0]])
        else:
            pred_box = expand_box(pred_box, self.origin_size, ratio=1.5)
        self.pred_box = floorcoord(pred_box)    # For it has been expanded
        
        prev_mask = cropimage(np.array(self.prev_mask), self.pred_box)
        track_image = cropimage(track_image, self.pred_box)

        # Resize to proper size
        ref_image, ref_mask, track_image, prev_mask = [resize(x, self.img_size) for x in [self.ref_image, self.ref_mask, track_image, prev_mask]]
        #prev_gmask = np.array(get_gb_image(prev_mask), np.float32)         # Gaussian previous mask
        prev_gmask = np.array(prev_mask, np.float32)

        DEBUG=False
        if DEBUG:
            cv2.imwrite('ref_image.jpg', ref_image[..., ::-1])
            cv2.imwrite('ref_mask.jpg', ref_mask*255)
            cv2.imwrite('prev_gmask.jpg', prev_gmask*255)
            cv2.imwrite('track_image.jpg', track_image[..., ::-1])

        # ToTensor
        ref_image, track_image = np.array(ref_image, dtype=np.float32), np.array(track_image, dtype=np.float32)  
        ref_mask, prev_gmask = np.array(ref_mask, dtype=np.uint8), np.array(prev_gmask, dtype=np.float32)
        
        # Normalize
        ref_image -= self.mean_value
        track_image -= self.mean_value
        ref_image, track_image = hwc2chw(ref_image), hwc2chw(track_image)
        prev_gmask = np.expand_dims(prev_gmask, axis=0)
        return ref_image.copy(), ref_mask.copy(), prev_gmask.copy(), track_image.copy()

    def __len__(self):
        return len(self.imglist[1:])



if __name__ == "__main__":
    #yvos_test = YVOSDataSet_Test()
    #yvos_test.reset("1bcd8a65de", 1)
    #yvos_test[0]
    test = DAVISDataSet_Ft()
    test.reset("bike-packing", 1)
    test[0]
    test = DAVISDataSet_Test()
    test.reset('bike-packing', '1')
    test[0]
    import pdb
    pdb.set_trace()














