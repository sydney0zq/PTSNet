#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
#
# Distributed under terms of the MIT license.

from logger.logger import setup_logger
from model.drsn import DRSN
from dataset.dataset_davis_test import DAVISDataSet_Test, DAVISDataSet_Ft, davis
from dataset.datautils import *
from torch.utils.data import DataLoader
import os, sys
import torch
import numpy as np
from torch import optim
import cv2
from PIL import Image
import warnings
cv2.ocl.setUseOpenCL(False)
warnings.filterwarnings('ignore')
import torch.backends.cudnn as cudnn
cudnn.enabled = cudnn.benchmark = True

class SeqResMemory:
    def __init__(self):
        self.seqdict = []
        self.prevbox = None
        self.img_size = [256, 256]
        self.mean_value = np.array((104, 117, 123), np.float)
        self.cnt = 1
    
    def reset(self):
        self.__init__()

    def enqueue(self, image, predmask):
        # Image and mask to tensor
        box = expand_box(get_mask_bbox(predmask), predmask.shape, ratio=1.5)
        if (box[2]-box[0]) <= 5 or (box[3] - box[1]) <= 5:
            if self.prevbox is None:
                box = (0, 0, predmask.shape[1], predmask.shape[0])
            else:
                box = expand_box(self.prevbox, predmask.shape, ratio=1.2)

        im_patch, mask_patch = cropimage(image, box), cropimage(predmask, box)
        im_patch, mask_patch = resize(im_patch, self.img_size), resize(mask_patch, self.img_size)

        #cv2.imwrite('DEBUG/im_patch_{}.jpg'.format(self.cnt), im_patch)
        #cv2.imwrite('DEBUG/mask_patch_{}.jpg'.format(self.cnt), mask_patch*255)
        #self.cnt += 1

        im_patch, mask_patch = np.array(im_patch, dtype=np.float32)-self.mean_value, np.array(mask_patch, dtype=np.uint8)
        im_patch = hwc2chw(im_patch)
        im_patch, mask_patch = unsqueezedim(im_patch), unsqueezedim(unsqueezedim(mask_patch))
        im_tensor = torch.from_numpy(im_patch).type(torch.FloatTensor)
        mask_tensor = torch.from_numpy(mask_patch).type(torch.FloatTensor)

        self.seqdict.append([im_tensor, mask_tensor])
        self.prevbox = box

    def fetch(self, index, cat=True):
        if cat is False:
            return self.seqdict[index]
        else:
            return torch.cat(self.seqdict[index], dim=1)


class SegWorker:
    def __init__(self, phase='test'):
        self.img_size = [256, 256]
        self.whole_model_path = "./snapshots/drsn_yvos_10w_davis_3p5w.pth"
        self.mask_local_home = "./result_davis_mask_local"
        self.mask_home = "./result_davis_mask"
        self.npy_home = "./result_davis_npy"
        self.phase = phase
        self.logger = setup_logger()
        self.ft_lr = None
        self.ft_iters = None
        self.ft_bs = None
        
    def get_dataloader_model(self):
        self.logger.info("Loading dataloader and model...")
        self.logger.info("Model comes from {}...".format(self.whole_model_path))
        testset = DAVISDataSet_Test(img_size=self.img_size)
        testloader = DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True, collate_fn=collate_fn)
        ftset = DAVISDataSet_Ft(img_size=self.img_size) 
        ftset.data_len = self.ft_iters * self.ft_bs
        ftloader = DataLoader(ftset, batch_size=self.ft_bs, shuffle=False, pin_memory=False, num_workers=self.ft_bs)
        model = DRSN()
        return testset, ftloader, testloader, model

    def ft(self, vdname, label_id):
        ft_args = {'ft_iters': self.ft_iters, 'ft_lr': self.ft_lr, 'ft_bs': self.ft_bs}
        self.logger.info ('Finetune args: {}'.format(ft_args))
        criterion = Criterion2()
        self.model.train()
        self.model.freeze_bn()
        self.model.cuda()
        criterion.cuda()
        optimizer = optim.Adam([{'params': filter(lambda p: p.requires_grad, self.model.parameters()), 'lr': self.ft_lr }], weight_decay=2e-4) 
        self.ftloader.dataset.reset(vdname, label_id)
        for i_iter, batch in enumerate(self.ftloader):
            ref_images, ref_masks, cand0_images, cand0_masks, cand1_images, cand1_masks, prev_gmasks, track_images, track_masks = batch
            track_masks = track_masks.long().cuda()

            input_rgb_mask = torch.cat([track_images.float(), prev_gmasks.float()], 1).cuda()
            cand0_rgb_mask = torch.cat([cand0_images.float(), cand0_masks.float()], 1).cuda()
            cand1_rgb_mask = torch.cat([cand1_images.float(), cand1_masks.float()], 1).cuda()
            ref_rgb_mask = torch.cat([ref_images.float(), ref_masks.float()], 1).cuda()

            optimizer.zero_grad()
            preds = self.model(ref_rgb_mask, cand0_rgb_mask, cand1_rgb_mask, input_rgb_mask)
            loss = criterion(preds, track_masks)
            loss.backward()
            optimizer.step()
            loss = loss.data.cpu().numpy()
            if i_iter % 10 == 0: self.logger.info('Finetune iter {} of {} completed, loss = {}'.format(i_iter, self.ft_iters, loss))


    def track_and_segment(self, vdname, label_id):
        self.model.freeze_bn()
        self.model.eval()

        seqresmem = SeqResMemory()
        self.logger.info("[Segmentation] Start to track and segment vd {} label {}".format(vdname, label_id))
        self.testset.reset(vdname, label_id)
        origin_size = self.testset.origin_size
        mask_local_home = os.path.join(self.mask_local_home, vdname, str(label_id))
        npy_home = os.path.join(self.npy_home, vdname, str(label_id))
        [os.makedirs(x, exist_ok=True) for x in [mask_local_home, npy_home]]
        init_mask_path = os.path.join(mask_local_home, "{}.png".format(get_img_str_idx(self.testset.imglist[0])))
        init_image, init_mask = self.testset.init_image, self.testset.init_mask
        cv2.imwrite(init_mask_path, init_mask*255)
        seqresmem.enqueue(init_image, init_mask)

        for index, batch in enumerate(self.testloader, 1):
            track_image_path = self.testset.imglist[index]
            track_frame_idx = get_img_str_idx(track_image_path)
            ref_image, ref_mask, prev_gmask, track_image = batch               
            ref_image_mask = torch.cat([ref_image, ref_mask], 1)
            pred_box = self.testset.pred_box
            track_image_mask = torch.cat([track_image, prev_gmask], 1)

            MEM_LENGTH = 5
            if index <= MEM_LENGTH:
                g_im1, g_im2 = ref_image_mask, ref_image_mask
            else:
                #g_idx1, g_idx2 = sorted(np.random.randint(index-MEM_LENGTH, index, 2))
                g_idx1, g_idx2 = index-4, index-2
                g_im1, g_im2 = seqresmem.fetch(g_idx1), seqresmem.fetch(g_idx2)

            res_cuda = self.model(ref_image_mask.cuda(), g_im1.cuda(), g_im2.cuda(),
                                  track_image_mask.cuda())
            res = res_cuda.data.cpu().numpy()[0]
            res = ndresize(res, res.shape[1:], (int(pred_box[3]-pred_box[1]), int(pred_box[2]-pred_box[0])))
            res_np = np.zeros((2, origin_size[0], origin_size[1]))
            res_np[:, pred_box[1]:pred_box[3], pred_box[0]:pred_box[2]] = res
            res_mask = np.argmax(res_np, axis=0)
            self.testset.prev_mask = res_mask

            # Add to reference pool
            seqresmem.enqueue(readimage(track_image_path), res_mask)

            track_mask_path = os.path.join(mask_local_home, "{}.png".format(track_frame_idx))
            cv2.imwrite(track_mask_path, res_mask*255)
            track_npy_path = os.path.join(npy_home, "{}.npy".format(track_frame_idx))
            np.save(track_npy_path, softmax(res_np)[1, :, :])

    def track_video(self, vdname): 
        self.testset, self.ftloader, self.testloader, self.model = self.get_dataloader_model()
        mask_home = os.path.join(self.mask_home, vdname)
        label_ids = davis.get_label_ids(vdname)
        os.makedirs(mask_home, exist_ok=True)
        for label_id in label_ids:
            self.model.init(self.whole_model_path)
            self.logger.info("[Segmentation] Start to ft vd {} label_id {}".format(vdname, label_id))
            self.ft(vdname, label_id)
            self.logger.info("[Segmentation] Start to track vd {} label_ids {}".format(vdname, label_ids))
            self.track_and_segment(vdname, label_id)
        img_list = davis.get_imglist(vdname)
        self.testset.reset(vdname, label_ids[0])
        origin_size = self.testset.origin_size
        for img_name in img_list[1:]:
            scores = []
            img_idx = get_img_str_idx(img_name)
            for label_id in label_ids:
                npy_path = os.path.join(self.npy_home, vdname, str(label_id), "{}.npy".format(img_idx))
                if os.path.exists(npy_path):
                    score = np.load(npy_path)
                else:
                    self.logger.warning ("{} nonexists...".format(npy_path))
                    score = np.zeros(origin_size)
                scores.append(score)
            bg_score = np.ones(origin_size) * 0.5
            scores = [bg_score] + scores
            score_all = np.stack(tuple(scores), axis = -1)
            label_pred = score_all.argmax(axis=2)
            label_pred = remap(label_pred, label_ids)
            ensumble_mask_path = os.path.join(mask_home, "{}.png".format(img_idx))
            #self.logger.info ("Final mask saved to {}".format(ensumble_mask_path))
            cv2.imwrite(ensumble_mask_path, label_pred)
        del_cmd = '/bin/rm -rf {}'.format(os.path.join(self.npy_home, vdname))
        self.logger.info ('Exec {}'.format(del_cmd))
        os.system(del_cmd)

if __name__ == "__main__":
    if len(sys.argv) == 4:
        dist_id = sys.argv[3]
    elif len(sys.argv) == 3:
        dist_id = 'val'
    else:
        assert False, "Params not enough..."
    dist_vdname = readvdnames(os.path.join('dist', 'dist_{}'.format(dist_id)))

    segworker = SegWorker()
    segworker.ft_lr = float(sys.argv[1])
    segworker.ft_iters = int(sys.argv[2])
    segworker.ft_bs = 16

    for vdname in dist_vdnames:
        segworker.track_video(vdname)

