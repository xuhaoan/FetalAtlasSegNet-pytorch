import os
import numpy as np
import torch


class AvgMeter(object):
    """
    Acc meter class, use the update to add the current acc
    and self.avg to get the avg acc
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dice_score(seg, gt, ratio=0.5):
    """
    function to calculate the dice score
    """
    seg = seg.flatten()
    gt = gt.flatten()
    seg[seg > ratio] = np.float32(1)
    seg[seg < ratio] = np.float32(0)
    dice = float(2 * (gt * seg).sum()) / float(gt.sum() + seg.sum())
    return dice


def multi_dice_score(seg, gt, ratio=0.5, num_class=9):
    """
    function to calculate the dice score
    """
    dice_score = []
    for label in range(1, num_class + 1):
        seg_label = (seg == label).astype(np.float32)
        gt_label = (gt == label).astype(np.float32)

        # Apply the ratio threshold
        seg_label[seg_label > ratio] = 1.0
        seg_label[seg_label <= ratio] = 0.0

        dice = 2.0 * (gt_label * seg_label).sum() / (gt_label.sum() + seg_label.sum())
        dice_score.append(dice)

    return sum(dice_score) / num_class


def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def mae_score(pred, gt):
    mae = float(abs(float(pred) - float(gt)))
    return mae