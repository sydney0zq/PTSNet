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
import random

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURR_DIR)
from davis_api import DAVIS
from datautils import *
import json
from data_augmentation import Augmentor


class MDNetRes_API:
    def __init__(self):
        self.resroot = os.path.join(CURR_DIR, "result_davis")
        self.davis = DAVIS()

    def get_seq_mdnet_res(self, vdname, label_id):
        mdnet_json = os.path.join(self.resroot, vdname, str(label_id), "result.json")
        if os.path.exists(mdnet_json):
            with open(mdnet_json, "r") as f:
                mdnet_preds = json.load(f)['res']
        else:
            init_box = self.davis.get_init_bbox(vdname, label_id)
            seq_len = len(self.davis.get_imglist(vdname))
            mdnet_preds = [init_box] * seq_len
        return np.array(mdnet_preds)


class DAVISDataSet(data.Dataset):
    def __init__(self, split='train', img_size=(256, 256), stage='stage1'):
        self.davis = DAVIS(split=split)
        self.seq_home = self.davis.dataroot
        self.img_size = img_size
        self.mean_value = np.array((104, 117, 123), np.float)
        self.data_len = len(self.davis)
        self.split = split
        self.auger = Augmentor(rand_rotate_angle=True, data_aug_flip=True, color_aug=True,
                               gaussian_blur=False, motion_blur=True)
        self.stage = stage
        self.mdnet_loader = MDNetRes_API()
        self.DEBUG = False

    def __getitem__(self, index):
        while True:
            seq_name = random.choice(self.davis.vdnames)
            label_id = random.choice(self.davis.get_label_ids(seq_name))
            imglist = self.davis.get_imglist(seq_name)
            annolist = self.davis.get_annolist(seq_name)
            seq_len = len(imglist)
            _ref_frame_idx, _track_frame_idx = np.random.choice(range(seq_len), 2, replace=True)
            ref_image_path, track_image_path = imglist[_ref_frame_idx], imglist[_track_frame_idx]
            ref_mask_path, track_mask_path = annolist[_ref_frame_idx], annolist[_track_frame_idx]
            ref_mask = readobjmask(ref_mask_path, int(label_id))
            track_mask = readobjmask(track_mask_path, int(label_id))
            if np.sum(ref_mask) >= 50:
                seq_mdnet_res = self.mdnet_loader.get_seq_mdnet_res(seq_name, label_id)
                ref_image, track_image = readimage(ref_image_path), readimage(track_image_path)
                ref_gd_box = get_mask_bbox(ref_mask)

                if self.stage == 'stage1':
                    if np.sum(track_mask) != 0:
                        track_gd_box = get_mask_bbox(track_mask)
                    else:
                        track_gd_box = ref_gd_box
                elif self.stage == 'stage2':
                    track_gd_box = get_mask_bbox(track_mask)
                    track_mdnet_box = seq_mdnet_res[_track_frame_idx]
                    track_mdnet_box = np.array(random.choice([track_gd_box, track_mdnet_box]))

                if _track_frame_idx == 0:
                    prev_frame_idx = _track_frame_idx
                else:
                    prev_frame_idx = _track_frame_idx-1
                prev_mask_path = annolist[prev_frame_idx]
                prev_mask = readobjmask(prev_mask_path, int(label_id))
                
                # Select cand ref ids
                _cand_frame_idx0, _cand_frame_idx1 = choose_rand_PQ(_track_frame_idx, seq_len)
                cand_frame_idx0, cand_frame_idx1 = imglist[_cand_frame_idx0], imglist[_cand_frame_idx1]
                cand_mask_idx0, cand_mask_idx1 = annolist[_cand_frame_idx0], annolist[_cand_frame_idx1]
                break

        # Crop necessary regions
        ref_box = expand_box(ref_gd_box, ref_mask.shape, ratio=1.5)
        ref_image, ref_mask = cropimage(ref_image, ref_box), cropimage(ref_mask, ref_box)

        if self.stage == 'stage1':
            track_box = shaking_bbox(track_image, track_gd_box, kContextFactor=2.0+round(random.random(), 1))
        elif self.stage == 'stage2':
            track_box = shaking_bbox(track_image, track_mdnet_box, kContextFactor=2.0+round(random.random(), 1))

        track_box = expand_box(track_box, track_image.shape, 1.2+min(0.3, round(random.random(), 1)))
        track_image, track_mask = cropimage(track_image, track_box), cropimage(track_mask, track_box)
        cand_image0, cand_image1 = readimage(cand_frame_idx0), readimage(cand_frame_idx1)
        cand_mask0, cand_mask1 = readobjmask(cand_mask_idx0, int(label_id)), readobjmask(cand_mask_idx1, int(label_id))
        cand_box0, cand_box1 = get_mask_bbox(cand_mask0), get_mask_bbox(cand_mask1)

        cand_box0 = shaking_bbox(cand_image0, cand_box0, kContextFactor=2.0+round(random.random(), 1))
        cand_box1 = shaking_bbox(cand_image1, cand_box1, kContextFactor=2.0+round(random.random(), 1))
        cand_box0 = expand_box(cand_box0, cand_mask0.shape, ratio=1.5)
        cand_box1 = expand_box(cand_box1, cand_mask1.shape, ratio=1.5)

        cand_image0, cand_mask0 = self.data_preprocess(cand_image0, cand_mask0, cand_box0, DEBUG_ID='cand0')
        cand_image1, cand_mask1 = self.data_preprocess(cand_image1, cand_mask1, cand_box1, DEBUG_ID='cand1')

        prev_mask = cropimage(prev_mask, track_box)

        # Resize to proper size
        ref_image, ref_mask = [resize(x, self.img_size) for x in [ref_image, ref_mask]]
        ref_image, track_image, ref_mask, track_mask, prev_mask = [resize(x, self.img_size) for x in [ref_image, track_image, ref_mask, track_mask, prev_mask]]

        # Augmentation
        if self.split == 'train': 
            track_image, track_mask = self.auger(track_image, track_mask)

        # ToTensor
        ref_image, track_image = np.array(ref_image, dtype=np.float32), np.array(track_image, dtype=np.float32)
        ref_mask, track_mask = np.array(ref_mask, dtype=np.uint8), np.array(track_mask, dtype=np.uint8)
        prev_mask = np.array(get_gb_image(prev_mask), np.float32)

        DEBUG=False
        if DEBUG:
            cv2.imwrite('ref_image.jpg', ref_image[:, :, ::-1])
            cv2.imwrite('track_image.jpg', track_image[:, :, ::-1])
            cv2.imwrite('ref_mask.jpg', ref_mask*255)
            cv2.imwrite('track_mask.jpg', track_mask*255)
            cv2.imwrite('prev_mask.jpg', prev_mask*255)

        ref_image -= self.mean_value
        track_image -= self.mean_value
        ref_image, track_image = hwc2chw(ref_image), hwc2chw(track_image)
        ref_mask, prev_mask = np.expand_dims(ref_mask, axis=0), np.expand_dims(prev_mask, axis=0)
        return ref_image.copy(), ref_mask.copy(), cand_image0.copy(), cand_mask0.copy(), \
               cand_image1.copy(), cand_mask1.copy(), prev_mask.copy(), track_image.copy(), track_mask.copy()
    
    def data_preprocess(self, image, mask, box, gt_bool=False, DEBUG_ID=None):
        if (box[2]-box[0]) * (box[3]-box[1]) <= 50 or (box[2]-box[0]) <= 5 or (box[3]-box[1]) <= 5:
            h, w = mask.shape
            box = (0, 0, w, h) 
        image, mask = cropimage(image, box), cropimage(mask, box)
        image, mask = resize(image, self.img_size), resize(mask, self.img_size)
        image, mask = np.array(image, dtype=np.float32), np.array(mask, np.uint8)

        if self.DEBUG and DEBUG_ID is not None:
            cv2.imwrite('{}_image.jpg'.format(DEBUG_ID), image[:, :, ::-1]) 
            cv2.imwrite('{}_mask.jpg'.format(DEBUG_ID), mask*255)

        image = hwc2chw(image - self.mean_value) 
        if gt_bool is False: mask = np.expand_dims(mask, axis=0) 
        return image.copy(), mask.copy()

    def __len__(self):
        return self.data_len
        
            


if __name__ == "__main__":
    #yvos_test = YVOSDataSet_Test()
    #yvos_test.reset("1bcd8a65de", 1)
    #yvos_test[0]
    test = DAVISDataSet()

    from torch.utils.data import DataLoader
    loader = DataLoader(test, batch_size=4)
    for data in loader:
        #test[0]
        import pdb
        pdb.set_trace()

