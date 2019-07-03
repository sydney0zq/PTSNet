#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <theodoruszq@gmail.com>

"""
YouTube-VOS Dataset.
"""

import os
import sys
from torch.utils import data
from torchvision import transforms
import random
import numpy as np
import torch
import cv2

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURR_DIR)
from yvos_api import YVOS
from datautils import *


class YVOSDataset(data.Dataset):
    def __init__(self, img_size=(256, 256)):
        self.yvos = YVOS(split='train')
        self.seq_home = self.yvos.dataroot
        self.img_size = img_size
        self.mean_value = np.array((104, 117, 123), np.float)
        self.data_len = 0
    
    def __getitem__(self, index):
        # Ensure training index and necessary data
        while True:
            seq_name = random.choice(self.yvos.vdnames)
            label_id = random.choice(self.yvos.get_label_ids(seq_name))
            frame_idxs = self.yvos.get_video_idxs(seq_name, label_id)
            seq_len = len(frame_idxs)
            _ref_frame_idx, _track_frame_idx = np.random.choice(range(seq_len), 2, replace=True)
            ref_frame_idx, track_frame_idx = frame_idxs[_ref_frame_idx], frame_idxs[_track_frame_idx]
            ref_mask = self.yvos.get_mask(seq_name, ref_frame_idx, label_id)

            if np.sum(ref_mask) >= 50:
                ref_image = self.yvos.get_image(seq_name, ref_frame_idx)
                track_image = self.yvos.get_image(seq_name, track_frame_idx)
                track_mask = self.yvos.get_mask(seq_name, track_frame_idx, label_id) 
                ref_gd_box, track_gd_box = get_mask_bbox(ref_mask), get_mask_bbox(track_mask)
                #if np.sum(track_mask) >= 50:
                #    track_gd_box = get_mask_bbox(track_mask)
                #else:
                #    track_gd_box = ref_gd_box

                if _track_frame_idx == 0:
                    prev_frame_idx = track_frame_idx
                else:
                    prev_frame_idx = frame_idxs[_track_frame_idx-1]
                prev_mask = self.yvos.get_mask(seq_name, prev_frame_idx, label_id)
                
                _p_frame_idx, _q_frame_idx = choose_rand_PQ(_track_frame_idx, seq_len)
                p_frame_idx, q_frame_idx = frame_idxs[_p_frame_idx], frame_idxs[_q_frame_idx]
                break
        # Prepare images and masks
        p_image, q_image = self.yvos.get_image(seq_name, p_frame_idx), self.yvos.get_image(seq_name, q_frame_idx)
        p_mask, q_mask = self.yvos.get_mask(seq_name, p_frame_idx, label_id), self.yvos.get_mask(seq_name, q_frame_idx, label_id)

        ref_image, ref_mask = self.data_preprocess(ref_image, ref_mask, box=ref_gd_box, shake_bool=False, expand_bool=True)
        p_image, p_mask = self.data_preprocess(p_image, p_mask, shake_bool=True, expand_bool=True)
        q_image, q_mask = self.data_preprocess(q_image, q_mask, shake_bool=True, expand_bool=True)

        track_box = shaking_bbox(track_image, track_gd_box, kContextFactor=2.0+round(random.random(), 1))
        track_box = expand_box(track_box, track_image.shape, 1.2+min(0.3, round(random.random(), 1)))
        track_image, track_mask = self.data_preprocess(track_image, track_mask, track_box, shake_bool=False, expand_bool=False, gt_bool=True)
        prev_mask = get_gb_image(resize(cropimage(prev_mask, track_box), self.img_size))
        prev_mask = np.expand_dims(np.array(prev_mask, np.float32), axis=0)
        
        return ref_image, ref_mask, p_image, p_mask, q_image, q_mask, prev_mask, track_image, track_mask

    def data_preprocess(self, image, mask, box=None, shake_bool=False, expand_bool=True, gt_bool=False):
        if box is not None:
            crop_box = box
        else:
            crop_box = get_mask_bbox(mask)
        # Shake
        if shake_bool is True: crop_box = shaking_bbox(image, crop_box, kContextFactor=2.0+round(random.random(), 1))
        # Expand
        if expand_bool is True: crop_box = expand_box(crop_box, mask.shape, ratio=1.5)
        image, mask = cropimage(image, crop_box), cropimage(mask, crop_box)
        image, mask = resize(image, self.img_size), resize(mask, self.img_size)
        image, mask = np.array(image, dtype=np.float32), np.array(mask, np.uint8)
        image = hwc2chw(image-self.mean_value)
        if gt_bool is False: mask = np.expand_dims(mask, axis=0)
        return image.copy(), mask.copy()

    def __len__(self):
        return self.data_len

    def visualize(self, index, vis_root):
        ref_image, ref_mask, p_image, p_mask, q_image, q_mask, prev_mask, track_image, track_mask = self.__getitem__(index)
        os.makedirs(vis_root, exist_ok=True)
        def inv_preprocess(image):
            image = chw2hwc(image)
            image = image + self.mean_value
            return image
        ref_image, p_image, q_image, track_image = [inv_preprocess(x)[:, :, ::-1] for x in [ref_image, p_image, q_image, track_image]]
        ref_mask, p_mask, q_mask, prev_mask, track_mask = [np.squeeze(x)*255 for x in [ref_mask, p_mask, q_mask, prev_mask, track_mask]]

        cv2.imwrite(os.path.join(vis_root, "ref_image.jpg"), ref_image)
        cv2.imwrite(os.path.join(vis_root, "ref_mask.png"), ref_mask)
        cv2.imwrite(os.path.join(vis_root, "p_image.jpg"), p_image)
        cv2.imwrite(os.path.join(vis_root, "p_mask.png"), p_mask)
        cv2.imwrite(os.path.join(vis_root, "q_image.jpg"), q_image)
        cv2.imwrite(os.path.join(vis_root, "q_mask.png"), q_mask)
        cv2.imwrite(os.path.join(vis_root, "prev_mask.png"), prev_mask)
        cv2.imwrite(os.path.join(vis_root, "track_image.png"), track_image)
        cv2.imwrite(os.path.join(vis_root, "track_mask.png"), track_mask)
        


if __name__ == "__main__":
    test = YVOSDataset()
    test.visualize(0, "DEBUG")
    import pdb
    pdb.set_trace()







