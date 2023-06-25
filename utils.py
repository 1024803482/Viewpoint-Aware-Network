import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


def structure_loss(y_true, y_pred, kernel_size=31, weights=5.0):
    """
    compute structure loss for binary segmentation map via torch
    """
    weit = 1.0 + weights * torch.abs(
        F.avg_pool2d(y_true, kernel_size=kernel_size, stride=1, padding=kernel_size // 2) - y_true)
    wbce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    y_pred = torch.sigmoid(y_pred)
    inter = ((y_pred * y_true) * weit).sum(dim=(2, 3))
    union = ((y_pred + y_true) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (1.0 * wbce + 1.0 * wiou).mean()


def mean_iou_np(y_true, y_pred, axes=(0, 1), **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, axes=(0, 1), **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)

    smooth = .001
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    return dice


def mean_seg_acc_np(y_true, y_pred):
    b, c, h, w = y_pred.shape
    return np.sum(y_true == y_pred) / (b * c * h * w)


def cal_cls_acc(pred, label):
    r = (torch.argmax(pred, dim=-1) == label).float()
    acc = torch.mean(r)
    return acc


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses) - self.num, 0):]))
