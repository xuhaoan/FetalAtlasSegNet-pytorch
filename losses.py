import os
import time

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn


def dice_coeff(pred, gt, eps=1e-5):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """
    pred = torch.sigmoid(pred)

    N = gt.size(0)
    pred_flat = pred.contiguous().view(N, -1)
    gt_flat = gt.contiguous().view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return loss.sum() / N


class SoftDiceLoss(torch.nn.Module):
    __name__ = 'dice_loss'

    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, y_pr, y_gt):
        return 1 - dice_coeff(y_pr, y_gt)


def sync_regularization(pred):
    GA = pred[1]
    pred = torch.sigmoid(pred[0])
    N = pred.size(0)
    sync_mae_loss = 0
    for i in range(N):
        seg_volume = pred[i, ...]
        seg_volume = (torch.sum(seg_volume) - 33300) / 245940

        GA_single = GA[i, 0, 0]
        GA_volume = (GA_single - 22.57) / 16.43

        mae_temp = torch.nn.SmoothL1Loss(reduction='mean')(seg_volume, GA_volume)
        sync_mae_loss += mae_temp
    sync_mae_loss /= N

    return sync_mae_loss


def loss_func(predict, label, weight=None):
    bce_loss = F.binary_cross_entropy_with_logits(predict, label, weight)
    dice_loss = SoftDiceLoss()(predict, label)
    return bce_loss + dice_loss


def loss_func_all(predict, label, weight=None):
    bce_loss = nn.CrossEntropyLoss()(predict, label)
    # dice_loss = SoftDiceLoss()(predict, label)
    # return bce_loss + dice_loss
    return bce_loss


def loss_mae(predict, label, GA, weight=None):
    bce_loss = F.binary_cross_entropy_with_logits(predict[0], label, weight)
    dice_loss = SoftDiceLoss()(predict[0], label)
    mae = torch.nn.SmoothL1Loss(reduction='mean')(predict[1], GA)
    return bce_loss + dice_loss + mae


def loss_mae_sync(predict, label, GA, weight=None):
    bce_loss = F.binary_cross_entropy_with_logits(predict[0], label, weight)
    dice_loss = SoftDiceLoss()(predict[0], label)
    mae = torch.nn.SmoothL1Loss(reduction='mean')(predict[1], GA)
    sync_loss = sync_regularization(predict)

    return bce_loss + dice_loss + mae + sync_loss