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
from dataset.dataset_davis_adapt import DAVISDataSet
import random
import timeit
import logging
from tensorboardX import SummaryWriter
from utils.util import decode_labels, inv_preprocess, decode_predictions
from utils.criterion import Criterion, Criterion2
from utils.model_init import init
from utils.learning_policy import adjust_learning_rate
from utils.parallel import SelfDataParallel, ModelDataParallel, CriterionDataParallel
from logger.logger import setup_logger
import warnings
warnings.filterwarnings('ignore')
CURR_DIR = os.path.dirname(__file__)
start = timeit.default_timer()

IMG_MEAN = np.array((104, 117, 123), dtype=np.float32)
sec2hm = lambda sec: [sec//3600, (sec%3600)//60]     # Seconds to hours and minutes

def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
            '--seg_model_path',
            type=str,
            required=False,
            default=osp.join(CURR_DIR, 'init_models/drsn_yvos_10w.pth'), 
            help='Model to initialize segmentation model')
    parser.add_argument(
            '--model_save_path',
            type=str,
            required=False,
            default="experiments", 
            help='Path to save models')
    parser.add_argument(
            '--im_size',
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
            '--lr_s1',
            type=float,
            required=False,
            default=2e-7,
            help='Learning rate in training s1')
    parser.add_argument(
            '--lr_s2',
            type=float,
            required=False,
            default=2e-8,
            help='Learning rate in training s2')
    parser.add_argument(
            '--steps_s1',
            type=int,
            required=False,
            default=25000,
            help='Learning steps in training')
    parser.add_argument(
            '--steps_s2',
            type=int,
            required=False,
            default=10000,
            help='Learning steps in training')
    parser.add_argument(
            '--save_iters',
            type=int,
            required=False,
            default=10000,
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

args = get_arguments()

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003

def validate(validloader, model, criterion, logger=None, n_batch=200):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for i_iter, batch in enumerate(validloader):
            if i_iter >= n_batch:
                break
            g_images, g_labels, pre_masks, images, labels = batch
            g_images = Variable(g_images.float().cuda())
            g_labels = Variable(g_labels.float().cuda())
            pre_masks = Variable(pre_masks.float().cuda())
            images = Variable(images.float().cuda())
            labels = Variable(labels.long().cuda())

            input_rgb_mask = torch.cat([images, pre_masks], 1)
            ref_rgb_mask = torch.cat([g_images, g_labels], 1)
            preds = model(ref_rgb_mask, input_rgb_mask)
            loss = criterion(preds, labels)
            loss = loss.data.cpu().numpy()
            if i_iter % 10 == 0:
                logger.info('Val iter {} / {} completed, loss = {}'.format(i_iter, n_batch, loss))
            valid_loss += loss
    valid_loss /= i_iter
    model.train()
    return valid_loss

def main():
    """Create the model and start the training."""
    logger = setup_logger()
    if os.environ['HOSTNAME'] != 'train119.hogpu.cc':
        writer = SummaryWriter(os.path.join("/job_tboard"))
    else:
        writer = SummaryWriter(args.model_save_path)
    logger.info(json.dumps(vars(args), indent=1))

    logger.info('Setting model...')
    model = DRSN()
    model.init(args.seg_model_path, stage='davis_train')
    model = torch.nn.DataParallel(model)
    model.train()
    model.float()
    #print(model)

    #model.eval() # use_global_stats = True
    # model = SelfDataParallel(model)
    # model.apply(set_bn_momentum)

    logger.info('Setting criterion...')
    criterion = Criterion2()       # For softmax
    #criterion = Criterion()         # For sigmoid
    #criterion = CriterionDataParallel(criterion)

    # Set CUDNN and GPU associated
    # gpu = args.gpu
    logger.info('Setting CUDNN...')
    cudnn.enabled = cudnn.benchmark = True
    model.cuda()    
    criterion.cuda()
    
    os.makedirs(args.model_save_path, exist_ok=True)
    [os.makedirs(os.path.join(args.model_save_path, x), exist_ok=True) for x in ['stage1', 'stage2']]

    trainset_s1 = DAVISDataSet(split='train', img_size=args.im_size, stage='stage1')
    trainset_s1.data_len = args.steps_s1 * args.batch_size
    trainset_s2 = DAVISDataSet(split='train', img_size=args.im_size, stage='stage2')
    trainset_s2.data_len = args.steps_s2 * args.batch_size
    trainloader_s1 = data.DataLoader(trainset_s1, batch_size=args.batch_size, shuffle=True, num_workers=args.batch_size//2+1, pin_memory=True)
    trainloader_s2 = data.DataLoader(trainset_s2, batch_size=args.batch_size, shuffle=True, num_workers=args.batch_size//2+1, pin_memory=True)

    #valset = DAVISDataSet(split='val', img_size=args.im_size)
    #valset.data_len = args.max_iters * args.batch_size
    #validloader = data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=args.batch_size//2+1, pin_memory=True)

    lr_s1 = args.lr_s1
    lr_s2 = args.lr_s2
    # Adam is better
    optimizer_s1 = optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': lr_s1 }], \
                                    lr=lr_s1, weight_decay=args.weight_decay)
    optimizer_s2 = optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': lr_s2 }], \
                                    lr=lr_s2, weight_decay=args.weight_decay)
    
    start_iter = 0
    for i_iter, batch in enumerate(trainloader_s1, start_iter):
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

        optimizer_s1.zero_grad()
        preds = model(ref_imasks, p_imasks, q_imasks, n_imasks)

        loss = criterion(preds, masks)
        loss.backward()
        optimizer_s1.step()
        loss = loss.data.cpu().numpy()

        if i_iter % 100 == 0:
            writer.add_scalar('MaskTrack_Loss/TrainLoss', loss, i_iter)

        #if i_iter % 1000 == 0 and i_iter != start_iter:
        #    logger.info('Start to do evaluation...')
        #    valid_loss = validate(validloader, model, criterion, logger=logger)
        #    writer.add_scalar('MaskTrack_Loss/ValidLoss', valid_loss, i_iter)

        #if i_iter % 500 == 0:
        #    g_images_inv = inv_preprocess(g_images, args.save_num_images, IMG_MEAN)
        #    images_inv = inv_preprocess(images, args.save_num_images, IMG_MEAN)
        #    g_labels_colors = decode_labels(g_labels, args.save_num_images, 2)
        #    labels_colors = decode_labels(labels, args.save_num_images, 2)
        #    if isinstance(preds, list):
        #        preds = preds[0]
        #    # probs = nn.functional.sigmoid(preds)
        #    preds_colors = decode_predictions(preds, args.save_num_images, 2)
        #    pre_masks_colors = decode_predictions(pre_masks, args.save_num_images, 2)
        #    for index, (img, lab) in enumerate(zip(images_inv, labels_colors)):
        #        writer.add_image('MaskTrack_CurImages/'+str(index), img, i_iter)
        #        writer.add_image('MaskTrack_CurLabels/'+str(index), lab, i_iter)
        #        writer.add_image('MaskTrack_RefImages/'+str(index), g_images_inv[index], i_iter)
        #        writer.add_image('MaskTrack_RefLabels/'+str(index), g_labels_colors[index], i_iter)
        #        writer.add_image('MaskTrack_CurPreds/'+str(index), preds_colors[index], i_iter)
        #        writer.add_image('MaskTrack_PreMasks/'+str(index), pre_masks_colors[index], i_iter)

        logger.info('Train adapt stage1 iter {} of {} completed, loss = {}'.format(i_iter, args.steps_s1, loss))

        if (i_iter+1) % args.save_iters == 0 or i_iter >= args.steps_s1-1:
            snapshot_fn = osp.join(args.model_save_path, 'stage1/drsn_davis_'+str(i_iter+1)+'.pth')
            logger.info("Snapshot {} dumped...".format(snapshot_fn))
            torch.save(model.state_dict(), snapshot_fn)     

    for i_iter, batch in enumerate(trainloader_s2, args.steps_s1):
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

        optimizer_s1.zero_grad()
        preds = model(ref_imasks, p_imasks, q_imasks, n_imasks)

        loss = criterion(preds, masks)
        loss.backward()
        optimizer_s2.step()
        loss = loss.data.cpu().numpy()

        if i_iter % 100 == 0:
            writer.add_scalar('MaskTrack_Loss/TrainLoss', loss, i_iter)

        #if i_iter % 1000 == 0 and i_iter != start_iter:
        #    logger.info('Start to do evaluation...')
        #    valid_loss = validate(validloader, model, criterion, logger=logger)
        #    writer.add_scalar('MaskTrack_Loss/ValidLoss', valid_loss, i_iter)

        #if i_iter % 500 == 0:
        #    g_images_inv = inv_preprocess(g_images, args.save_num_images, IMG_MEAN)
        #    images_inv = inv_preprocess(images, args.save_num_images, IMG_MEAN)
        #    g_labels_colors = decode_labels(g_labels, args.save_num_images, 2)
        #    labels_colors = decode_labels(labels, args.save_num_images, 2)
        #    if isinstance(preds, list):
        #        preds = preds[0]
        #    # probs = nn.functional.sigmoid(preds)
        #    preds_colors = decode_predictions(preds, args.save_num_images, 2)
        #    pre_masks_colors = decode_predictions(pre_masks, args.save_num_images, 2)
        #    for index, (img, lab) in enumerate(zip(images_inv, labels_colors)):
        #        writer.add_image('MaskTrack_CurImages/'+str(index), img, i_iter)
        #        writer.add_image('MaskTrack_CurLabels/'+str(index), lab, i_iter)
        #        writer.add_image('MaskTrack_RefImages/'+str(index), g_images_inv[index], i_iter)
        #        writer.add_image('MaskTrack_RefLabels/'+str(index), g_labels_colors[index], i_iter)
        #        writer.add_image('MaskTrack_CurPreds/'+str(index), preds_colors[index], i_iter)
        #        writer.add_image('MaskTrack_PreMasks/'+str(index), pre_masks_colors[index], i_iter)

        logger.info('Train adapt stage2 iter {} of {} completed, loss = {}'.format(i_iter, args.steps_s2+args.steps_s1, loss))

        if (i_iter+1) % args.save_iters == 0 or i_iter >= args.steps_s2+args.steps_s1-1:
            snapshot_fn = osp.join(args.model_save_path, 'stage2/drsn_davis_'+str(i_iter+1)+'.pth')
            logger.info("Snapshot {} dumped...".format(snapshot_fn))
            torch.save(model.state_dict(), snapshot_fn)
            
    end = timeit.default_timer()
    total_h, total_m = sec2hm(end - start)
    logger.info('The whole training costs {}h {}m...'.format(total_h, total_m))

if __name__ == '__main__':
    main()
