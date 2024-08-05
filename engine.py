import math
import time
import matplotlib.pyplot as plt

import torch
import numpy as np
from utils import AverageValueMeter

def train_one_epoch_supervised(epoch, model, criterion, args, train_loader, avg_meters, logger):
    start_time = time.time()
    model.train()
    for bidx, data in enumerate(train_loader):
        step = bidx + len(train_loader) * epoch
        bsize = data['set'].size(0)
        preds = model(data['set'].cuda(), data['set_mask'].cuda())
        # prepare categorical targets for cross entropy loss
        target_classes, targets = np.unique(data['mid'], return_inverse=True)
        loss = criterion(preds, torch.tensor(targets).squeeze().cuda())
        acc = torch.sum(torch.argmax(preds, dim=1) ==  torch.tensor(targets).squeeze().cuda())/preds.shape[0]
        model.optimizer.zero_grad()
        model.backward(loss)
        # compute gradient norm
        total_norm = 0.
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        if logger is not None and total_norm > 1000:
            logger.add_scalar('grad_norm', total_norm, step)
        if args.max_grad_threshold is not None:
            if total_norm < args.max_grad_threshold:
                model.optimizer.step()
        else:
            model.optimizer.step()

        # Only main process writes logs.
        avg_meters['acc_avg_meter'].update(acc.detach().item(), bsize)
        avg_meters['loss_avg_meter'].update(loss.detach().item(), bsize)

        if step % args.log_freq == 0:
            duration = time.time() - start_time
            start_time = time.time()

            print("[Rank %d] Epoch %d Batch [%2d/%2d] Time [%3.2fs] Acc %2.5f Loss %2.5f"
                  % (args.local_rank, epoch, bidx, len(train_loader), duration, acc.detach().item(),
                     loss.detach().item()))

            if logger is not None:
                logger.add_scalar('train x-ent loss (step)', loss.detach().item(), step)
                logger.add_scalar('train acc (step)', acc.detach().item(), step)
                logger.add_scalar('grad_norm', total_norm, step)

        # assert after logging and optimizing to sync subprocesses
        loss_finite = math.isfinite(loss.detach().item())
        assert loss_finite

    if logger is not None:
        logger.add_scalar('train acc (epoch)', avg_meters['acc_avg_meter'].avg, epoch)
        logger.add_scalar('train x-ent loss (epoch)', avg_meters['loss_avg_meter'].avg, epoch)
        avg_meters['acc_avg_meter'].reset()
        avg_meters['loss_avg_meter'].reset()

def validate_supervised(model, args, val_loader, epoch, criterion, logger):
    model.eval()
    start_time = time.time()
    acc_meter = AverageValueMeter()
    loss_meter = AverageValueMeter()
    for bidx, data in enumerate(val_loader):
        bsize = data['set'].size(0)
        preds = model(data['set'].cuda(), data['set_mask'].cuda())
        #preds = output['predictions']
        target_classes, targets = np.unique(data['mid'], return_inverse=True)
        loss = criterion(preds, torch.tensor(targets).squeeze().cuda())
        acc = torch.sum(torch.argmax(preds, dim=1) ==  torch.tensor(targets).squeeze().cuda())/len(torch.argmax(preds, dim=1) ==  torch.tensor(targets).squeeze().cuda())

        """
        # compute gradient norm
        total_norm = 0.
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        if logger is not None and total_norm > 1000:
            logger.add_scalar('grad_norm', total_norm, step)
        """

        # Only main process writes logs.
        acc_meter.update(acc.detach().item(), bsize)
        loss_meter.update(loss.detach().item(), bsize)

        # assert after logging and optimizing to sync subprocesses
        loss_finite = math.isfinite(loss.detach().item())
        assert loss_finite

    # log val set stats for this epoch
    duration = time.time() - start_time
    print("[Rank %d] <VAL> Epoch %d Batch [%2d/%2d] Time [%3.2fs] Acc %2.5f Loss %2.5f"
            % (args.local_rank, epoch, bidx, len(val_loader), duration, acc_meter.avg, loss_meter.avg))
    if logger is not None:
        logger.add_scalar('val acc (epoch)', acc_meter.avg, epoch)
        logger.add_scalar('val x-ent loss (epoch)', loss_meter.avg, epoch)

    return {'val_acc': acc_meter.avg, 'val_loss': loss_meter.avg}