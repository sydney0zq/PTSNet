#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
#
# Distributed under terms of the MIT license.

import numpy as np
def overlay_mask(image, mask):
    """ image is a HxWxC np.array, mask is a HxW np.array """
    label_colours = [(0,0,0)
                    # 0=background
                    ,(0,256,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                    # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                    ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                    # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                    ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                    # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                    ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                    # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
    indices = np.unique(mask)
    for cls_index in indices:
        print (cls_index)
        if cls_index != 0:
            mask_index = mask == cls_index
            image[mask_index, :] = image[mask_index, :] * 0.5 + np.array(label_colours)[cls_index]* 0.5
    return image

from PIL import Image
import glob
import os
import cv2
if __name__ == "__main__":
    yvosroot = "./dataset/DAVIS/trainval/JPEGImages/480p"
    maskroot = "./result_davis_mask"
    rendroot = "./result_visualize"
    listFile = "./dataset/DAVIS/trainval/ImageSets/2017/val.txt"
    with open(listFile, 'r') as f:
        fds = [line.strip() for line in f]

    for vdname in vdnames:
        print ("on processing {}".format(vdname))
        #if os.path.exists(os.path.join(maskroot, vdname)) is False:
        #    continue
        imlist = sorted(glob.glob(os.path.join(yvosroot, vdname, "*.jpg")))
        masklist = sorted(glob.glob(os.path.join(maskroot, vdname, "*.png")))
        #assert(len(imlist) == len(masklist))
        os.makedirs(os.path.join(rendroot, vdname), exist_ok=True)
        for maskname in masklist:
            frame_idx = maskname.split('/')[-1][:-4]
            imname = os.path.join(yvosroot, vdname, "{}.jpg".format(frame_idx))
            print (imname)
            print (maskname)
            im, mask = cv2.imread(imname), np.array(Image.open(maskname))
            rendim = overlay_mask(im, mask)
            cv2.imwrite(os.path.join(rendroot, vdname, imname.split('/')[-1]), rendim)

