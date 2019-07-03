#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <theodoruszq@gmail.com>

""" Object Tracking Network coupled with Object Proposal Network. """
import argparse
import numpy as np
import os
import sys
import time
import json
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn

import os.path as osp
def add_path(paths):
    for path in paths:
        if path not in sys.path: sys.path.insert(0, path)
this_dir = osp.abspath(osp.dirname(__file__))
lib_path = osp.join(this_dir, 'maskrcnn', 'lib') # Add lib to PYTHONPATH
dep_path = osp.join(this_dir, '..')
add_path([lib_path, dep_path])
from otn_modules.sample_generator import *
from data_prov import *
from otn_modules.model import *
from options import *
from gen_config import *
from otn_modules.utils import *

## Object Proposal Network deps ##
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test_det import im_detect_all
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)
np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')


def convert_from_cls_format(cls_boxes):
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    return boxes

class Detector:
    def __init__(self):
        cfg_from_file("maskrcnn/configs/baselines/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x.yaml")
        cfg.RESNETS.IMAGENET_PRETRAINED = False  # Don't need to load imagenet pretrained weights
        assert_and_infer_cfg()

        maskRCNN = Generalized_RCNN()
        maskRCNN.cuda()
        pretrained_path = "maskrcnn/data/X-152-32x8d-FPN-IN5k.pkl"
        print("loading detectron weights %s" % pretrained_path)
        load_detectron_weight(maskRCNN, pretrained_path)
        maskRCNN.eval()         # Note this step

        self.maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                     minibatch=True, device_ids=[0])  # only support single GPU
    def detect(self, impath):
        im = cv2.imread(impath)
        assert im is not None
        cls_boxes = im_detect_all(self.maskRCNN, im)
        return cls_boxes

################################################

def forward_samples(model, image, samples, out_layer='conv3'):
    model.eval()
    extractor = RegionExtractor(image, samples, opts['img_size'], opts['padding'], opts['batch_test'])
    for i, regions in enumerate(extractor):
        regions = Variable(regions)
        if opts['use_gpu']:
            regions = regions.cuda()
        feat = model(regions, out_layer=out_layer)
        if i==0:
            feats = feat.data.clone()
        else:
            feats = torch.cat((feats,feat.data.clone()),0)
    return feats


def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    model.train()
    
    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while(len(pos_idx) < batch_pos*maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while(len(neg_idx) < batch_neg_cand*maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for iter in range(maxiter):
        # select pos idx
        pos_next = pos_pointer+batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next
        # select neg idx
        neg_next = neg_pointer+batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next
        # create batch
        batch_pos_feats = Variable(pos_feats.index_select(0, pos_cur_idx))
        batch_neg_feats = Variable(neg_feats.index_select(0, neg_cur_idx))
        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0,batch_neg_cand,batch_test):
                end = min(start+batch_test,batch_neg_cand)
                score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start==0:
                    neg_cand_score = score.data[:,1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:,1].clone()),0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
            model.train()
        # forward
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)
        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
        optimizer.step()
        #print "Iter %d, Loss %.4f" % (iter, loss.data[0])

def sort_box(bbox_clsdet, bbox_rect=None, pos_range=[0.7, 1], neg_range=[0, 0.3], sampler=None, gen_pos_num=None, gen_neg_num=None):
    #bbox_rect = otb2cross(bbox_rect)
    bbox_det = cross2otb(bbox_clsdet[:, :4])
    print ("[INFO] Box number from Object Proposal Network(ori): ", len(bbox_det))

    # First compute all overlap_arr > 0.7 take them out
    overlap_arr = overlap_ratio(bbox_det, bbox_rect)
    idx = np.ones(len(overlap_arr), dtype=bool)
    pos_arr_idx = idx * ((overlap_arr >= pos_range[0]) * (overlap_arr <= pos_range[1]))
    neg_arr_idx = idx * ((overlap_arr >= neg_range[0]) * (overlap_arr <= neg_range[1]))

    pos_box, neg_box = bbox_det[pos_arr_idx, :], bbox_det[neg_arr_idx, :]
    print ("[INFO] Pos/Neg box number from maskrcnn(overlap>0.7/<0.3): {}/{}".format(len(pos_box), len(neg_box)))
    if len(pos_box) != 0:
        tmp_box = np.vstack([pos_box, bbox_rect.reshape(1, 4)])
        num_each = gen_pos_num // len(tmp_box)
        if num_each >= 3:       # /5
            for i in range(len(tmp_box)):
                pos_box_ = gen_samples(sampler, tmp_box[i, :].reshape(4), num_each, pos_range)
                if len(pos_box.shape) == 1:
                    pos_box = np.vstack([pos_box_, pos_box.reshape(1, 4)])
                else:
                    pos_box = np.vstack([pos_box_, pos_box])
    else:
        print ('[WARN] Object Proposal Network generates no overlap > 0.7 boxes....')
        print (bbox_rect)
        pos_box = gen_samples(sampler, bbox_rect, gen_pos_num, pos_range)
    
    # Filter out w / h == 0 boxes
    pass_posid = (pos_box[:, 2] > 1) * (pos_box[:, 3] > 1)
    pass_negid = (neg_box[:, 2] > 1) * (neg_box[:, 3] > 1)
    pos_box = pos_box[pass_posid, :]
    neg_box = neg_box[pass_negid, :]

    while len(pos_box) < gen_pos_num:
        pos_box = np.vstack([pos_box, pos_box])
    pos_box = pos_box[:gen_pos_num, :]
    while len(neg_box) < gen_neg_num:
        neg_box = np.vstack([neg_box, neg_box])
    neg_box = neg_box[:gen_neg_num, :]
    
    return pos_box, neg_box

def sort_box_(bbox_clsdet, bbox_rect, overlap=0.3):
    bbox_det = cross2otb(bbox_clsdet[:, :4])
    overlap_arr = overlap_ratio(bbox_det, bbox_rect)
    idx = np.ones(len(overlap_arr), dtype=bool)
    arr_idx = idx * (overlap_arr > overlap)
    return bbox_det[arr_idx, :]

## Main Function ##

def run_mdnet(img_list, init_bbox, gt=None, savefig_dir=''):
    detector = Detector()

    # Compute how many object and get their init boxes
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list),4))
    result[0] = target_bbox
    result_top5 = np.zeros((len(img_list), 5, 5))
    result_top5[0, :, :] = [1, target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3]]

    if target_bbox[2] * target_bbox[3] <= 50:
        print ("[WARN] Too small object to track...")
        for i in range(1, len(img_list)):
            result[i] = result[0]
            result_top5[i, :, :] = result_top5[0]
        return result, result_top5

    # Init model
    model = MDNet(opts['model_path'])
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])
    
    # Init criterion and optimizer 
    criterion = BinaryLoss()
    init_optimizer = set_optimizer(model, opts['lr_init'])
    update_optimizer = set_optimizer(model, opts['lr_update'])

    tic = time.time()
    # Load first image
    image = Image.open(img_list[0]).convert('RGB')
    imsize = image.size

    # Draw pos/neg samples out to initialize our model
    start = time.time()
    init_detect_clsbox = detector.detect(img_list[0])
    end = time.time()
    print ("detector:", end-start)
    init_sampler = SampleGenerator('gaussian', image.size, 0.1, 1.1)
    det_posbox, det_negbox = sort_box(init_detect_clsbox, target_bbox, sampler=init_sampler, gen_pos_num=opts['n_pos_init'], gen_neg_num=opts['n_neg_init'])
    print ("[INFO] init pos/neg box number: {}/{}".format(det_posbox.shape[0], det_negbox.shape[0]))
    pos_examples, neg_examples = det_posbox, det_negbox

    pos_examples = np.random.permutation(pos_examples)
    neg_examples = np.random.permutation(neg_examples)

    # Extract pos/neg features
    pos_feats = forward_samples(model, image, pos_examples)
    neg_feats = forward_samples(model, image, neg_examples)
    feat_dim = pos_feats.size(-1)
    init_pos_feats = pos_feats
    init_neg_feats = neg_feats

    # Initial training
    train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])
    print ("[INFO] init train finished...")
    
    # Init pos/neg features for update
    pos_feats_all = [pos_feats[:opts['n_pos_update']]]
    neg_feats_all = [neg_feats[:opts['n_neg_update']]]
    
    spf_total = time.time()-tic

    # Display
    savefig = savefig_dir != ''
    if savefig: 
        dpi = 80.0
        figsize = (image.size[0]/dpi, image.size[1]/dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image, aspect='normal')
        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0,:2]),gt[0,2],gt[0,3], 
                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)
        
        rect = plt.Rectangle(tuple(result[0,:2]),result[0,2],result[0,3], 
                linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)
        print (savefig_dir)
        fig.savefig(os.path.join(savefig_dir, '00000.jpg'),dpi=dpi)
        plt.close()
    
    s = time.time()
    # Main loop / Tracking
    success = True
    success_sampler = SampleGenerator('gaussian', image.size, 0.1, 1.1)
    fail_sampler = SampleGenerator('gaussian', image.size, 0.1, 1.1)
    for i in range(1,len(img_list)):
        tic = time.time()
        image = Image.open(img_list[i]).convert('RGB')

        # Estimate target bbox and select some of them (overlap > 0.3)
        detect_clsbox = detector.detect(img_list[i])
        samples_box = sort_box_(detect_clsbox, target_bbox, overlap=0.3)
        samples = samples_box
        
        if success:
            samples_box = sort_box_(detect_clsbox, target_bbox, overlap=0.3)
            samples = samples_box
            if len(samples) < 5:
                samples_ = gen_samples(success_sampler, target_bbox, 5)
                samples = np.vstack([samples_, samples])
        else:
            samples_box = sort_box_(detect_clsbox, target_bbox, overlap=0.2)
            samples = samples_box
            if len(samples) < 5:
                samples_ = gen_samples(fail_sampler, target_bbox, 5)
                samples = np.vstack([samples_, samples])
            
        sample_scores = forward_samples(model, image, samples, out_layer='fc6')
        top_scores, top_idx = sample_scores[:,1].topk(5)
        top_idx = top_idx.cpu().numpy()
        top_box = samples[top_idx]
        target_score = top_scores.mean()
        target_bbox = samples[top_idx].mean(axis=0)

        # draw out boxes whose score bigger than 0.5
        prob_idx = np.ones(len(sample_scores[:, 1].cpu().numpy()), dtype=bool)
        prob_idx = prob_idx * (sample_scores[:, 1].cpu().numpy() >= 0.5)
        prob_box = samples[prob_idx]

        success = target_score > opts['success_thr']
        if success:
            result_top5[i, :, 0] = top_scores
            result_top5[i, :, 1:] = top_box[:, :]
        else:
            result_top5[i, :, 0] = -1
            result_top5[i, :, 1:] = result[i-1]

        
        if not success: target_bbox = result[i-1]
        
        # Save result
        result[i] = target_bbox

        # Data collect
        if success:
            #detect_clsbox = detector.detect(img_list[i])
            det_posbox, det_negbox = sort_box(detect_clsbox, target_bbox, sampler=init_sampler, gen_pos_num=opts['n_pos_update'], gen_neg_num=opts['n_neg_update'])
            pos_examples, neg_examples = det_posbox, det_negbox
            print ("[INFO] loop pos/neg box number: {}/{}".format(det_posbox.shape[0], det_negbox.shape[0]))

            pos_examples = np.random.permutation(pos_examples)
            neg_examples = np.random.permutation(neg_examples)

            # Extract pos/neg features
            pos_feats = forward_samples(model, image, pos_examples)
            neg_feats = forward_samples(model, image, neg_examples)
            pos_feats_all.append(pos_feats)
            neg_feats_all.append(neg_feats)
            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(5,len(pos_feats_all))
            pos_data = torch.stack(pos_feats_all[-nframes:],0).view(-1,feat_dim)
            neg_data = torch.stack(neg_feats_all,0).view(-1,feat_dim)
            train(model, criterion, update_optimizer, pos_data, neg_data, 10)
        
        # Long term update
        elif i % opts['long_interval'] == 0:
            nframes = min(opts['n_frames_short'],len(pos_feats_all))
            pos_data = torch.stack(pos_feats_all[-nframes:],0).view(-1,feat_dim)
            neg_data = torch.stack(neg_feats_all,0).view(-1,feat_dim)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])
        
        spf = time.time()-tic
        spf_total += spf

        # Display
        if savefig:
            dpi = 80.0
            figsize = (image.size[0]/dpi, image.size[1]/dpi)

            fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            im = ax.imshow(image, aspect='normal')

            if gt is not None:
                gt_rect = plt.Rectangle(tuple(gt[i,:2]),gt[i,2],gt[i,3], 
                        linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
                ax.add_patch(gt_rect)

            # draw prob box
            for i_p in range(len(prob_box)):
                prob_rect = plt.Rectangle(tuple(prob_box[i_p, :2]),prob_box[i_p, 2],prob_box[i_p, 3], 
                        linewidth=1, edgecolor="#87CEFA", zorder=1, fill=False, linestyle='dashed')
                ax.add_patch(prob_rect)
            rect = plt.Rectangle(tuple(result[i,:2]),result[i,2],result[i,3], 
                    linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
            ax.add_patch(rect)

            for topid in range(5):
                top_rect = plt.Rectangle(tuple(top_box[topid, :2]),top_box[topid, 2],top_box[topid, 3],
                        linewidth=2, edgecolor="#ffb3a7", zorder=1, fill=False, linestyle='solid') 
                ax.add_patch(top_rect) 

            fig.savefig(os.path.join(savefig_dir,'%05d.jpg'%(i)),dpi=dpi)
            plt.close()

        if gt is None:
            print ("Frame %d/%d, Score %.3f, Time %.3f" % \
                (i, len(img_list), target_score, spf))
        else:
            print ("Frame %d/%d, Overlap %.3f, Score %.3f, Time %.3f" % \
                (i, len(img_list), overlap_ratio(gt[i],result[i])[0], target_score, spf))
    print(len(img_list), "costs {}".format(time.time()-s))

    fps = len(img_list) / spf_total
    return result, result_top5, fps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='', help='input seq')
    parser.add_argument('-l', '--label_id', default=1)

    args = parser.parse_args()
    
    # Generate sequence config
    img_list, init_bbox, savefig_dir, result_path = gen_config(args.seq, args.label_id)

    # Run tracker
    result, result_top5, fps = run_mdnet(img_list, init_bbox, gt=None, savefig_dir=savefig_dir)

    # Save result
    res = {}
    res['res'] = result.round().tolist()
    res['type'] = 'rect'
    res['fps'] = fps
    json.dump(res, open(os.path.join(result_path, "result.json"), 'w'), indent=2)

