import os
import random
import numpy as np
from pathlib import Path
from pprint import pprint
import time

import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

import torch
from models.criterion import ChamferCriterion

class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def pad(x, x_mask, max_size):
    if x.size(1) < max_size:
        pad_size = max_size - x.size(1)
        pad = torch.ones(x.size(0), pad_size, x.size(2)).to(x.device) * float('inf')
        pad_mask = torch.ones(x.size(0), pad_size).bool().to(x.device)
        x, x_mask = torch.cat((x, pad), dim=1), torch.cat((x_mask, pad_mask), dim=1)
    else:
        UserWarning(f"pad: {x.size(1)} >= {max_size}")
    return x, x_mask


def extend_batch(b, b_mask, x, x_mask):
    if b is None:
        return x, x_mask
    if b.size(1) >= x.size(1):
        x, x_mask = pad(x, x_mask, b.size(1))
    else:
        b, b_mask = pad(b, b_mask, x.size(1))
    return torch.cat((b, x), dim=0), torch.cat((b_mask, x_mask), dim=0)


def save(model, optimizer, scheduler, epoch, path):
    d = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(d, path)


def resume(path, model, optimizer=None, scheduler=None, strict=True):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model'], strict=strict)
    start_epoch = ckpt['epoch']
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler'])
    return model, optimizer, scheduler, start_epoch
