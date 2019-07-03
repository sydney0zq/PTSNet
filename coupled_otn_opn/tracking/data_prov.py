import sys
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

sys.path.insert(0, "..")

from otn_modules.utils import *

class RegionExtractor():
    def __init__(self, image, samples, crop_size, padding, batch_size, shuffle=False):

        self.image = np.asarray(image)
        self.samples = samples
        self.crop_size = crop_size
        self.padding = padding
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.index = np.arange(len(samples))
        self.pointer = 0

        self.mean = self.image.mean(0).mean(0).astype('float32')

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.samples):
            self.pointer = 0
            raise StopIteration
        else:
            next_pointer = min(self.pointer + self.batch_size, len(self.samples))
            index = self.index[self.pointer:next_pointer]
            self.pointer = next_pointer

            regions = self.extract_regions(index)
            regions = torch.from_numpy(regions)
            return regions
    next = __next__

    def extract_regions(self, index):
        regions = np.zeros((len(index),self.crop_size,self.crop_size,3),dtype='uint8')
        for i, sample in enumerate(self.samples[index]):
            regions[i] = crop_image(self.image, sample, self.crop_size, self.padding)

        regions = regions.transpose(0,3,1,2).astype('float32')
        regions = regions - 128.
        return regions


class SegRegionDataset():

    def __init__(self, image, label, samples, crop_size, max_iters, is_mirror=True):

        self.image = np.asarray(image)
        self.label = np.asarray(label)
        self.samples = [sample for sample in samples]
        self.crop_size = crop_size
        self.mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
        self.is_mirror = is_mirror
        # repeat the training list
        if not max_iters==None:
            self.samples  = self.samples  * int(np.ceil(float(max_iters) / len(self.samples)))

    def __getitem__(self, index):
        img_h, img_w = self.label.shape
        image = Image.fromarray(self.image)
        label = Image.fromarray(self.label)
        
        box = self.samples[index]
        sample_box = otb2cross(box).tolist()

        image = image.crop(sample_box)
        label = label.crop(sample_box)

        image = image.resize(self.crop_size, Image.BILINEAR)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.uint8)
        image -= self.mean

        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy()

    def __len__(self):
        return len(self.samples)

def SegRegionExtractor(image, samples, crop_size, padding):
    IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

    regions = np.zeros((samples.shape[0], crop_size, crop_size,3), dtype='float32')
    for i, sample in enumerate(samples):
        regions[i] = crop_image(image, sample, crop_size, padding)

    regions = regions - IMG_MEAN
    regions = regions.transpose(0,3,1,2).astype('float32')

    return regions
