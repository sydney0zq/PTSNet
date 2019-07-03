#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
#
# Distributed under terms of the MIT license.

import os
import glob
import numpy as np
from PIL import Image
import sys
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURR_DIR)

readvdnames = lambda x: open(x).read().rstrip().split('\n')
readmask = lambda x: np.array(Image.open(x))
get_img_int_idx = lambda x: int(x.split('/')[-1][:-4])
from datautils import get_mask_bbox, cross2otb

pathmap = {'train':'trainval', 'val': 'trainval', 'dev': 'testdev', 'test': 'testchallenge'}

class DAVIS:
    def __init__(self, split='train'):
        self.split = split
        self.dataroot = os.path.join(CURR_DIR, "DAVIS/{}".format(pathmap[split]))
        self.vdnames = readvdnames(os.path.join(self.dataroot, 'ImageSets/2017/{}.txt'.format(self.split)))

    """ Get video's all ids """
    def get_label_ids(self, vdname):
        label_ids = np.unique(readmask(os.path.join(self.dataroot, 'Annotations/480p/{}/00000.png'.format(vdname))))[1:]
        return sorted(list(label_ids))

    """ Get video's all frame paths """
    def get_imglist(self, vdname):
        imglist = glob.glob(os.path.join(self.dataroot, "JPEGImages/480p/{}/*.jpg".format(vdname)))
        return sorted(imglist)

    def get_annolist(self, vdname):
        annolist = glob.glob(os.path.join(self.dataroot, "Annotations/480p/{}/*.png".format(vdname)))
        return sorted(annolist)
    
    def get_init_bbox(self, vdname, label_id):
        init_mask = np.array(Image.open(os.path.join(self.dataroot, "Annotations/480p/{}/00000.png".format(vdname)))) == int(label_id)
        init_bbox = np.array(get_mask_bbox(init_mask))
        init_bbox = cross2otb(init_bbox)
        init_bbox = [int(x) for x in init_bbox]
        return init_bbox

    def __len__(self):
        cnt = 0
        for vdname in self.vdnames:
            for label_id in self.get_label_ids(vdname):
                seq_len = len(self.get_imglist(vdname))
                cnt += seq_len
        return cnt


if __name__ == "__main__":
    test = DAVIS()
    #for vdname in test.vdnames:
    #    print (vdname)

    import pdb
    pdb.set_trace()

