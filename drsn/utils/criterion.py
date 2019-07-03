import torch.nn as nn
# import encoding.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
import scipy.ndimage as nd

def balanced_binary_cross_entropy(output, label):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """
    assert len(label.size()) == 3
    label = torch.unsqueeze(label, 1)
    labels_pos = (label == 1.0).float()
    labels_neg = (label == 0.0).float()
    
    num_labels_pos = torch.sum(labels_pos)
    num_labels_neg = torch.sum(labels_neg)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = (output >= 0).float()
    #print ("output dimension: ", output.size())
    #print ("labels dimension: ", labels.size())
    #print ("output * (labels - output_gt_zero) dimension: ", (output*(labels-output_gt_zero)).size())
    #print ("torch.exp(output - 2 * output * output_gt_zero) dimension: ", (torch.exp(output - 2 * output * output_gt_zero)).size())
    #print ("torch.log(1 + torch.exp(output - 2 * output * output_gt_zero) dimension: ", (1 + torch.exp(output - 2 * output * output_gt_zero)).size())
    loss_val = output * (labels_pos - output_gt_zero) - torch.log( \
            1 + torch.exp(output - 2 * output * output_gt_zero))

    loss_pos = -torch.sum(labels_pos*loss_val)
    loss_neg = -torch.sum(labels_neg*loss_val)

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    #return final_loss / label.size(0)      # Size averaged
    return final_loss                       # No size averaged


class Criterion(nn.Module):
    def __init__(self, ignore_index=255):
        super(Criterion, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = balanced_binary_cross_entropy

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        # scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear')
        loss = self.criterion(preds, target)
        return loss

class Criterion2(nn.Module):
    def __init__(self, ignore_index=255):
        super(Criterion2, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(size_average=False, ignore_index=ignore_index) 
        # self.criterion = OhemCrossEntropy2d(ignore_index, 0.7, 0.05)

    def forward(self, preds, target):
        #h, w = target.size(1), target.size(2)
        # scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear')
        loss = self.criterion(preds, target)

        return loss
