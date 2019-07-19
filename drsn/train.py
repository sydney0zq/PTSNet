#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <theodoruszq@gmail.com>
""" DRSN training code on YouTube-VOS. """

import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import json
import os.path as osp
from model.drsn import DRSN
from dataset.dataset_yvos import YVOSDataset
import random
import time
import logging
from tensorboardX import SummaryWriter
from utils.criterion import Criterion, Criterion2
from logger.logger import setup_logger
import warnings
warnings.filterwarnings('ignore')
CURR_DIR = os.path.dirname(__file__)

IMG_MEAN = np.array((104, 117, 123), dtype=np.float32)
sec2hm = lambda sec: [sec//3600, (sec%3600)//60]     # Seconds to hours and minutes
cudnn.enabled = cudnn.benchmark = True

def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
            '--init_model_path',
            type=str,
            required=False,
            default=osp.join(CURR_DIR, 'init_models/resnet50-19c8e357.pth'),
            help='Model to initialize segmentation model')
    parser.add_argument(
            '--model_save_path',
            type=str,
            required=False,
            default="experiments", 
            help='Path to save models')
    parser.add_argument(
            '--img_size',
            nargs=2, type=int,
            required = False,
            default=[256, 256],
            help='Input image size')
    parser.add_argument(
            '--batch_size',
            type=int,
            required = False,
            default=8,
            help='Batch size in training')
    parser.add_argument(
            '--save_num_images',
            type=int,
            required=False,
            default=2,
            help='Number of samples to be saved')
    parser.add_argument(
            '--decayat',
            type=int,
            required=False,
            default=30000,
            help='Number of samples to be saved')
    parser.add_argument(
            '--learning_rate',
            type=float,
            nargs='+',
            required=False,
            default=[1e-5, 1e-6],
            help='Learning rate in training')
    parser.add_argument(
            '--learning_policy',
            type=str,
            required=False,
            choices=['step','poly','constant'],
            default="step",
            help='Pick one learning policy from: step, poly, constant')
    parser.add_argument(
            '--max_iters',
            type=int,
            required=False,
            default=300000,
            help='Training iterations')
    parser.add_argument(
            '--save_iters',
            type=int,
            required=False,
            default=20000,
            help='Save model per this number of iterations')
    parser.add_argument(
            '--weight_decay',
            type=float,
            required=False,
            default=0.0002,
            help='Learning weight decay for L2 norm')
    parser.add_argument(
            '--power',
            type=float,
            required=False,
            default=0.9,
            help='Learning power for poly learning policy')
    return parser.parse_args()

def main():
    args = get_arguments()
    start = time.time()

    """Create the model and start the training."""
    logger = setup_logger()
    writer = SummaryWriter(args.model_save_path)
    logger.info(json.dumps(vars(args), indent=1))

    logger.info('Setting model...')
    model = DRSN()
    model.init(args.init_model_path, "yvos_train")
    model.train()
    model.float()
    model = torch.nn.DataParallel(model.cuda())
    #print(model)
    logger.info('Setting criterion...')
    criterion = Criterion2().cuda()       # For softmax

    os.makedirs(args.model_save_path, exist_ok=True)

    trainset = YVOSDataset(args.img_size)
    trainset.data_len = args.max_iters * args.batch_size
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=32, pin_memory=True)

    learning_rate = args.learning_rate[0]
    # Adam is better
    optimizer = optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': learning_rate }], \
                                    lr=learning_rate, weight_decay=args.weight_decay)
    
    for i_iter, batch in enumerate(trainloader):
        ref_images, ref_masks, p_images, p_masks, q_images, q_masks, pre_masks, images, masks = batch
        ref_images, p_images, q_images, images = ref_images.float().cuda(), p_images.float().cuda(), \
                                                 q_images.float().cuda(), images.float().cuda()
        ref_masks, p_masks, q_masks, pre_masks, masks = ref_masks.float().cuda(), p_masks.float().cuda(), \
                                                        q_masks.float().cuda(), pre_masks.float().cuda(), \
                                                        masks.long().cuda()
        ref_imasks = torch.cat([ref_images, ref_masks], 1)
        p_imasks = torch.cat([p_images, p_masks], 1)
        q_imasks = torch.cat([q_images, q_masks], 1)
        n_imasks = torch.cat([images, pre_masks], 1)

        optimizer.zero_grad()
        #adjust_learning_rate(optimizer, i_iter, args)
        #preds = model(ref_rgb_mask, cand_rgb_mask0, cand_rgb_mask1, input_rgb_mask)
        preds = model(ref_imasks, p_imasks, q_imasks, n_imasks)

        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        loss = loss.data.cpu().numpy()

        if i_iter == args.decayat:
            optimizer.param_groups[0]['lr'] = learning_rate * 0.1
            
        if i_iter % 200 == 0:
            writer.add_scalar('MaskTrack_LearningRate', optimizer.param_groups[0]['lr'], i_iter)
            writer.add_scalar('MaskTrack_Loss/TrainLoss', loss, i_iter)

        #if i_iter % 500 == 0:
        #    g_images_inv = inv_preprocess(ref_images, args.save_num_images, IMG_MEAN)
        #    images_inv = inv_preprocess(images, args.save_num_images, IMG_MEAN)
        #    g_labels_colors = decode_labels(ref_masks, args.save_num_images, 2)
        #    labels_colors = decode_labels(masks, args.save_num_images, 2)
        #    if isinstance(preds, list):
        #        preds = preds[0]
        #    preds_colors = decode_predictions(preds, args.save_num_images, 2)
        #    pre_masks_colors = decode_predictions(pre_masks, args.save_num_images, 2)
        #    for index, (img, lab) in enumerate(zip(images_inv, labels_colors)):
        #        writer.add_image('MaskTrack_CurImages/'+str(index), img, i_iter)
        #        writer.add_image('MaskTrack_CurLabels/'+str(index), lab, i_iter)
        #        writer.add_image('MaskTrack_RefImages/'+str(index), g_images_inv[index], i_iter)
        #        writer.add_image('MaskTrack_RefLabels/'+str(index), g_labels_colors[index], i_iter)
        #        writer.add_image('MaskTrack_CurPreds/'+str(index), preds_colors[index], i_iter)
        #        writer.add_image('MaskTrack_PreMasks/'+str(index), pre_masks_colors[index], i_iter)

        logger.info('Train iter {} of {} completed, loss = {}'.format(i_iter, args.max_iters, loss))

        if (i_iter+1) % args.save_iters == 0 or i_iter >= args.max_iters-1:
            snapshot_fn = osp.join(args.model_save_path, 'drsn_'+str(i_iter+1)+'.pth')
            logger.info("Snapshot {} dumped...".format(snapshot_fn))
            torch.save(model.state_dict(), snapshot_fn)     
            
    end = time.time()
    total_h, total_m = sec2hm(end - start)
    logger.info('The whole training costs {}h {}m...'.format(total_h, total_m))

if __name__ == '__main__':
    main()
