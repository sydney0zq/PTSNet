#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@train119.hogpu.cc>
#
# Distributed under terms of the MIT license.

"""
Appearance augmentation module.
"""

import numpy as np
from PIL import Image, ImageEnhance
import cv2
import random
import math
PI = 3.1415926

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[:2])/2)
    rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
    angle_r = float(angle) / 180 * PI
    result = cv2.warpAffine(image, rot_mat, image.shape[:2], flags=cv2.INTER_NEAREST)
    return result

def genaratePsf(degree, angle):
    # 这里生成任意角度的运动模糊kernel的矩阵，degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree  
    anchor = (0, 0)
    psf1 = M.copy()
    if angle<90 and angle>0:
        psf1=np.fliplr(psf1)
        anchor=(psf1.shape[1]-1, 0)
    elif angle>-90 and angle<0:#同理：往右下角移动
        psf1=np.flipud(psf1)
        psf1=np.fliplr(psf1)
        anchor=(psf1.shape[1]-1,psf1.shape[0]-1)
    elif angle<-90:#同理：往左下角移动
        psf1=np.flipud(psf1)
        anchor=(0,psf1.shape[0]-1)
    return motion_blur_kernel, anchor

def brightness_contrast_aug(im, brightness_range=(0.8, 1.3), contrast_range=(0.8, 1.3)):
    enhancer = ImageEnhance.Brightness(im)
    factor = np.random.uniform(brightness_range[0], brightness_range[1], 1)
    im = enhancer.enhance(factor)
    enhancer = ImageEnhance.Contrast(im)
    factor = np.random.uniform(contrast_range[0], contrast_range[1], 1)
    im = enhancer.enhance(factor)
    return im


class Augmentor:
    def __init__(self, rand_rotate_angle=False, data_aug_flip=False, color_aug=False, gaussian_blur=False, motion_blur=False):
        self.rand_rotate_angle = rand_rotate_angle
        self.data_aug_flip = data_aug_flip
        self.color_aug = color_aug
        self.gaussian_blur = gaussian_blur
        self.motion_blur = motion_blur

    def __call__(self, im, label):
        TOPIL = False if type(im).__module__ == np.__name__ else True

        if TOPIL is False: im, label = Image.fromarray(im.astype('uint8')), Image.fromarray(label.astype('uint8'))

        if self.rand_rotate_angle:
            if random.random() > 0.7:
                angle = (random.random() - 0.5) * 2 * self.rand_rotate_angle
                im = Image.fromarray(rotate_image(np.array(im), angle))
                label = Image.fromarray(rotate_image(np.array(label), angle))

        if self.data_aug_flip:
            if random.random() > 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)

        if self.color_aug:
            if random.random() > 0.5:
                im = brightness_contrast_aug(im)
        
        if self.gaussian_blur:
            if random.random() > 0.9:
                while True:
                    ks = random.randint(3, 10)
                    if ks % 2 != 0: break
                im = cv2.GaussianBlur(np.array(im), (ks, ks), 0)
                label = np.array(cv2.GaussianBlur(np.array(label), (ks, ks), 0) > 0.5, dtype='uint8')

        if self.motion_blur:
            if random.random() > 0.9:
                degree, angle = random.randint(10, 25), random.randint(0, 360)
                motion_blur_kernel, anchor = genaratePsf(degree, angle)
                
                #print (kernel, anchor)
                im = cv2.filter2D(np.array(im), -1, motion_blur_kernel, anchor=anchor)
                cv2.normalize(im, im, 0, 255, cv2.NORM_MINMAX)
                label = np.array(cv2.filter2D(np.array(label), -1, motion_blur_kernel, anchor=anchor) > 0.5, dtype='uint8')
        
        if TOPIL is False: im, label = np.array(im), np.array(label)
        return im, label

if __name__ == '__main__':
    im = Image.open('00000.jpg')
    label = Image.open('00000.png')
    im = im.crop((200, 200, 600, 600)).resize((256, 256))
    label = label.crop((200, 200, 600, 600)).resize((256, 256))
    im, label = np.array(im), np.array(np.array(label) == 255, dtype='uint8')

    cv2.imwrite('im.jpg', im)
    cv2.imwrite('label.jpg', label*255)
    #auger = Augmenter(rand_rotate_angle=10, data_aug_flip=False)
    #auger = Augmenter(gaussian_blur=True)
    auger = Augmentor(motion_blur=True)
    #while True:
    #    print ('once')
    #    im, label = auger(im, label)
    im, label = auger(im, label)
    cv2.imwrite('im_aug.jpg', im)
    cv2.imwrite('label_aug.jpg', label*255)



