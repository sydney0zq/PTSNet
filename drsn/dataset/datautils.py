#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <theodoruszq@gmail.com>

"""
Utils for loading dataset.
"""

import numpy as np
import random
import math
from PIL import Image
import scipy.ndimage as nd

######################### Box Processing ######################### 
""" Expand bounding box in height and width axes respectively.
    :box: In format of (minx, miny, maxx, maxy);
    :masksize: In format of (h, w);
    :ratio: Equal to expand_w/ori_w.
"""
def expand_box(box, masksize, ratio=1.2):
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    w,  h  =  box[2]-box[0]   ,  box[3]-box[1]
    new_minx = max(0, cx-ratio*w/2.0)
    new_maxx = min(masksize[1], cx+ratio*w/2.0)
    new_miny = max(0, cy-ratio*h/2.0)
    new_maxy = min(masksize[0], cy+ratio*h/2.0)
    return (new_minx, new_miny, new_maxx, new_maxy)

""" Transform box type. OTB means (minx, miny, w, h) while cross means (minx, miny, maxx, maxy). 
    :box: Nx4 data structure in OTB format
"""
def otb2cross(box):
    box = np.array(box)
    if len(box.shape) == 1:
        minx, miny = box[0], box[1]
        w, h = box[2], box[3]
        return np.array([minx, miny, minx+w, miny+h])
    elif len(box.shape) == 2:
        minx, miny = box[:, 0], box[:, 1]
        w, h = box[:, 2], box[:, 3]
        return np.stack([minx, miny, minx+w, miny+h], axis=1)
    else:
        print ("otb2cross function's input shape is {}".format(box.shape))
        assert False, "Plase check your otb2cross function's input..."
def cross2otb(box):
    if len(box.shape) == 1:
        minx, miny = box[0], box[1]
        maxx, maxy = box[2], box[3]
        return np.array([minx, miny, maxx-minx, maxy-miny])
    elif len(box.shape) == 2:
        minx, miny = box[:, 0], box[:, 1]
        maxx, maxy = box[:, 2], box[:, 3]
        w, h = maxx-minx, maxy-miny
        return np.stack([minx, miny, w, h], axis=1)
    else:
        print ("cross2otb function's input shape is {}".format(box.shape))
        assert False, "Plase check your cross2otb function's input..."

""" Shake the box to simulate the property of OTN. Code borrows from:
    https://github.com/nrupatunga/PY-GOTURN
    :image: Only the shape of image is used.
    :box: Original box.
    :bbparam: Modulate adjustment parameters. You can refer to the origin paper.
"""
def sample_rand_uniform():
    RAND_MAX = 2147483647
    return (random.randint(0, RAND_MAX) + 1) * 1.0 / (RAND_MAX + 2)
def sample_exp_two_sides(lambda_):
    RAND_MAX = 2147483647
    randnum = random.randint(0, RAND_MAX)
    if (randnum % 2) == 0:
        pos_or_neg = 1
    else:
        pos_or_neg = -1
    rand_uniform = sample_rand_uniform()
    return math.log(rand_uniform) / (lambda_ * pos_or_neg)
def shaking_bbox(image, bbox,
          bbparam={ 'lambda_scale_frac': 15, 'lambda_shift_frac': 5, 'min_scale': -0.4, 'max_scale': 0.4 }, 
          kContextFactor=1.5, LaplaceBool=True):
    lambda_shift_frac, lambda_scale_frac = bbparam['lambda_shift_frac'], bbparam['lambda_scale_frac']
    min_scale, max_scale = bbparam['min_scale'], bbparam['max_scale']
    width, height = bbox[2] - bbox[0], bbox[3]-bbox[1]
    center_x, center_y = bbox[0] + width/2., bbox[1] + height/2.
    kMaxNumTries = 10

    new_width = -1
    num_tries_width = 0
    while ((new_width < 0) or (new_width > image.shape[1] - 1)) and (num_tries_width < kMaxNumTries):
        width_scale_factor = max(min_scale, min(max_scale, sample_exp_two_sides(lambda_scale_frac)))
        new_width = width * (1 + width_scale_factor)
        new_width = max(1.0, min((image.shape[1] - 1), new_width))
        num_tries_width = num_tries_width + 1

    new_height = -1
    num_tries_height = 0
    while ((new_height < 0) or (new_height > image.shape[0] - 1)) and (num_tries_height < kMaxNumTries):
        height_scale_factor = max(min_scale, min(max_scale, sample_exp_two_sides(lambda_scale_frac)))
        new_height = height * (1 + height_scale_factor)
        new_height = max(1.0, min((image.shape[0] - 1), new_height))
        num_tries_height = num_tries_height + 1

    first_time_x = True
    new_center_x = -1
    num_tries_x = 0
    while ((first_time_x or (new_center_x < center_x - width * kContextFactor / 2)
           or (new_center_x > center_x + width * kContextFactor / 2)
           or ((new_center_x - new_width / 2) < 0)
           or ((new_center_x + new_width / 2) > image.shape[1]))
           and (num_tries_x < kMaxNumTries)):
        new_x_temp = center_x + width * sample_exp_two_sides(lambda_shift_frac)
        new_center_x = min(image.shape[1] - new_width / 2, max(new_width / 2, new_x_temp))
        first_time_x = False
        num_tries_x = num_tries_x + 1

    first_time_y = True
    new_center_y = -1
    num_tries_y = 0
    while ((first_time_y or (new_center_y < center_y - height * kContextFactor / 2)
           or (new_center_y > center_y + height * kContextFactor / 2)
           or ((new_center_y - new_height / 2) < 0)
           or ((new_center_y + new_height / 2) > image.shape[0]))
           and (num_tries_y < kMaxNumTries)):
        new_y_temp = center_y + height * sample_exp_two_sides(lambda_shift_frac)
        new_center_y = min(image.shape[0] - new_height / 2, max(new_height / 2, new_y_temp))
        first_time_y = False
        num_tries_y = num_tries_y + 1
    box =  [new_center_x - new_width/2., new_center_y - new_height/2.,
            new_center_x + new_width/2., new_center_y + new_height/2.]
    return box

""" Overlap (IoU) betweem two box sets. The box should be in OTB format.
    :rect1: Nx4 numpy array boxes, in (minx, miny, w, h) format.
    :rect2: Nx4 numpy array boxes, in (minx, miny, w, h) format.
"""
def overlap_ratio(rect1, rect2):
    # rect1 and rect2 should in otb format
    if rect1.ndim==1: rect1 = rect1[None,:]
    if rect2.ndim==1: rect2 = rect2[None,:]
    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0] + rect1[:,2], rect2[:,0] + rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1] + rect1[:,3], rect2[:,1] + rect2[:,3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:,2] * rect1[:,3] + rect2[:,2] * rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou

""" Resize image/mask by scipy, size should be in (h, w). """
def ndresize(image, in_size, out_size):
    if len(image.shape) == 3:
        image_resize = nd.zoom(image.astype('float32'),
                        (1.0, out_size[0]/float(in_size[0]), out_size[1]/float(in_size[1])))
    elif len(image.shape) == 4:
        image_resize = nd.zoom(image.astype('float32'),
                        (1.0, 1.0, out_size[0]/float(in_size[0]), out_size[1]/float(in_size[1])))
    else:
        image_resize = nd.zoom(image.astype('float32'),
                        (out_size[0]/float(in_size[0]), out_size[1]/float(in_size[1])))
    return image_resize

""" The input should be in 1D or 2D numpy array. """
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


""" Only used for YouTube-VOS inference. """
def remap(mask, label_ids):
    remap_mask = np.zeros(mask.shape)
    label_ids = sorted([int(x) for x in label_ids])
    if label_ids[0] != 0: label_ids = [0] + label_ids
    mask_ids = [* range(len(label_ids))]
    for mask_id, label_id in zip(mask_ids, label_ids):
        #print ("maskid {} to label_id {}".format(mask_id, label_id))
        remap_mask[mask==mask_id] = label_id
    return remap_mask


######################### Mask Processing ######################### 
""" Compute the bounding box enclosing the mask. Borrowed from: 
        https://github.com/linjieyangsc/video_seg
    :m: Binary mask in numpy array;
    :border_pixels: Border length.
"""
def get_mask_bbox(m, border_pixels=0):
    if not np.any(m):
        return (0, 0, m.shape[1], m.shape[0])
    rows, cols = np.any(m, axis=1), np.any(m, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    h, w = m.shape
    ymin = max(0, ymin - border_pixels)
    ymax = min(h-1, ymax + border_pixels)
    xmin = max(0, xmin - border_pixels)
    xmax = min(w-1, xmax + border_pixels)
    return (xmin, ymin, xmax, ymax)

""" Assistant for get_gb_image. """
def compute_robust_moments(binary_image, isotropic=False):
    index = np.nonzero(binary_image)
    points = np.asarray(index).astype(np.float32)
    if points.shape[1] == 0:
        return np.array([-1.0,-1.0],dtype=np.float32), \
            np.array([-1.0,-1.0],dtype=np.float32)
    points = np.transpose(points)
    points[:,[0,1]] = points[:,[1,0]]
    center = np.median(points, axis=0)
    if isotropic:
        diff = np.linalg.norm(points - center, axis=1)
        mad = np.median(diff)
        mad = np.array([mad,mad])
    else:
        diff = np.absolute(points - center)
        mad = np.median(diff, axis=0)
    std_dev = 1.4826*mad
    std_dev = np.maximum(std_dev, [5.0, 5.0])
    return center, std_dev

""" Compute a Gaussian float mask referring the origin mask, indicating the coarse position.
    Borrowed from: https://github.com/linjieyangsc/video_seg
    :label: Binary mask;
    :center/std perturb: Perturb variables;
    :blank_prob: Probility of an all zero mask.
"""
def get_gb_image(label, center_perturb=0.2, std_perturb=0.4, blank_prob=0):
    label = np.array(label)
    if not np.any(label) or random.random() < blank_prob:
        return np.zeros((label.shape))
    center, std = compute_robust_moments(label)
    center_p_ratio = np.random.uniform(-center_perturb, center_perturb, 2)
    center_p = center_p_ratio * std + center
    std_p_ratio = np.random.uniform(1.0 / (1 + std_perturb), 1.0 + std_perturb, 2)
    std_p = std_p_ratio * std
    h, w = label.shape
    x = np.arange(0, w)
    y = np.arange(0, h)
    nx, ny = np.meshgrid(x,y)
    coords = np.concatenate((nx[...,np.newaxis], ny[...,np.newaxis]), axis = 2)
    normalizer = 0.5 /(std_p * std_p)
    D = np.sum((coords - center_p) ** 2 * normalizer, axis=2)
    D = np.exp(-D)
    D = np.clip(D, 0, 1)
    return D


######################### Image Processing ######################### 
readmask = lambda x: np.array(Image.open(x)).astype(np.uint8)
readimage = lambda x: np.array(Image.open(x).convert('RGB'))
readobjmask = lambda x, obj_id: (np.array(Image.open(x)) == int(obj_id)).astype(np.uint8)
hwc2chw = lambda x: x.transpose((2, 0, 1))
chw2hwc = lambda x: x.transpose((1, 2, 0))
topil = lambda x: Image.fromarray(x.astype(np.uint8)) if isinstance(x, np.ndarray) else x
resize = lambda x, size: np.array(topil(x).resize(size), x.dtype)
get_img_str_idx = lambda x: x.split('/')[-1][:-4]
floorcoord = lambda x: [math.floor(c) for c in x]
unsqueezedim = lambda x: np.expand_dims(x, axis=0)
def cropimage(image, box):
    """ image: ndarray/PIL; box: cross format """
    image = np.array(image)
    minx, miny, maxx, maxy = [int(x) for x in box]
    width, height = max(0, maxx - minx), max(0, maxy - miny)
    assert (not(width == 0 or height == 0)), \
            "Nonsense cropbox: ({}, {}, {}, {})...".format(minx, miny, maxx, maxy)

    roi_left = max(0, min(minx, image.shape[1]))
    roi_right = min(image.shape[1], max(0, maxx))
    roi_top = max(0, min(miny, image.shape[0]))
    roi_bottom = min(image.shape[0], max(0, maxy))
    roi_width, roi_height = roi_right-roi_left, roi_bottom-roi_top
    pad_x, pad_y = max(0, 0-minx), max(0, 0-miny)
    
    if len(image.shape) == 3:
        padimage = np.zeros((height, width, 3))
        padimage[pad_y:pad_y+roi_height, pad_x:pad_x+roi_width, :] = image[roi_top:roi_bottom, roi_left:roi_right, :]
    elif len(image.shape) == 2:
        padimage = np.zeros((height, width))
        padimage[pad_y:pad_y+roi_height, pad_x:pad_x+roi_width] = image[roi_top:roi_bottom, roi_left:roi_right]

    return padimage

import torch
def collate_fn(data):
    # data[0] is a tuple with 4 elements
    if data[0] is None:
        return None
    else:
        ret = []
        for item in data[0]:
            if len(item.shape) == 2:
                item = torch.from_numpy(item).unsqueeze(0).unsqueeze(1)
            elif len(item.shape) == 3:
                item = torch.from_numpy(item).unsqueeze(0)
            else:
                assert False, "Collate fn throws up an error of datatype not compatible..."
            ret.append(item.type(torch.FloatTensor))
        return ret


if __name__ == "__main__":
    x = np.zeros((800, 800, 3)) + 0.5
    #x[100:200, 200:300, 0] = 128
    y = cropimage(x, [0, 0, 100, 100])
    print (np.unique(y))
    exit()



######################### MISC Processing ######################### 
""" Choose random P and Q as dynamic reference.
    :N: The current frame index.
    :seq_len: The length of a sequence.
"""
def choose_rand_PQ(N, seq_len):
    inverse = True if random.random() >= 0.5 else False
    if inverse is False:
        P, Q = sorted(np.random.choice(range(max(0, N-4), max(1, N)), 2, replace=True))
    else:
        P, Q = sorted(np.random.choice(range(min(seq_len-1, N+1), min(seq_len, N+4)), 2, replace=True), reverse=True)
    return P, Q





