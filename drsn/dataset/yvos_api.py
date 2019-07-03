#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
#
# Distributed under terms of the MIT license.

import json
import os
import glob
import numpy as np
from PIL import Image

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
readimage = lambda x: np.array(Image.open(x).convert('RGB'))
readmask = lambda x: np.array(Image.open(x)).astype(np.uint8)
readobjmask = lambda x, obj_id: (np.array(Image.open(x)) == int(obj_id)).astype(np.uint8)

class YVOS:
    def __init__(self, split=None):
        self.split = split
        assert (split is not None), "Split is not assigned..."
        self.dataroot = os.path.join(CURR_DIR, "yvos", split)
        #self.dataroot = os.path.join(CURR_DIR, "yvos/valid")
        metafile = os.path.join(self.dataroot, "meta.json")
        self.meta = json.load(open(metafile))['videos']
        self.vdnames = sorted(self.meta.keys())

    def get_label_ids(self, vdname):
        label_ids = sorted(self.meta[vdname]['objects'].keys())
        return list(label_ids)

    def get_video_idxs(self, vdname, label_id):
        if self.split not in ['test_all_frames']:
            return self.meta[vdname]['objects'][str(label_id)]['frames']
        else:
            return self.meta[vdname]['objects'][str(label_id)]

    def get_image(self, vdname, frame_idx):
        image_path = os.path.join(self.dataroot, 'JPEGImages', vdname, "{:05d}.jpg".format(int(frame_idx)))
        image = readimage(image_path)
        return image

    def get_mask(self, vdname, frame_idx, label_id=None):
        mask_path = os.path.join(self.dataroot, 'Annotations', vdname, "{:05d}.png".format(int(frame_idx)))
        if label_id is None:
            mask = readmask(mask_path)
        else:
            mask = readobjmask(mask_path, label_id)
        return mask

    # Video first annotation appeared index
    def get_video_start_frame_idx(self, vdname):
        label_ids = self.get_objects(vdname)
        start_frame_ids = [int(self.get_video_obj(vdname, x)[0]) for x in label_ids]
        return min(start_frame_ids)

    def get_full_imglist(self, vdname):
        label_ids = self.get_objects(vdname)
        start_frame_idx = self.get_video_start_frame_idx(vdname)
        imglist = []
        for label_id in label_ids:
            if start_frame_idx == int(self.get_video_obj(vdname, label_id)[0]):
                imglist = [os.path.join(self.dataroot, vdname, "{}.jpg".format(x)) for x in self.get_video_obj(vdname, label_id)]
                return imglist
                
    def get_sys_imglist(self, vdname):
        imglist = sorted(glob.glob(os.path.join(self.dataroot, "JPEGImages", vdname, "*.jpg")))
        return imglist

if __name__ == "__main__":
    test = YVOS(split='valid_all_frames')
    for vdname in test.vdnames:
        label_ids = test.get_objects(vdname)
        if label_ids == ['1'] or label_ids == ['1', '2', '3'] or label_ids == ['1', '2']:
            pass
        else:
            print (label_ids)

    import pdb
    pdb.set_trace()

