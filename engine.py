import math
import time
import matplotlib.pyplot as plt

import torch
import numpy as np
from utils import validate_reconstruct, validate_reconstruct_l2, validate_sample, \
    visualize_reconstruct, visualize_sample, visualize_mix, visualize_interpolate, AverageValueMeter

def train_one_epoch(epoch, model, criterion, optimizer, args, train_loader, avg_meters, logger):
    start_time = time.time()
    model.train()
    criterion.train()
    beta = None
    for bidx, data in enumerate(train_loader):
        step = bidx + len(train_loader) * epoch
        gt, gt_mask = data['set'], data['set_mask']
        gt = gt.cuda(non_blocking=True)
        gt_mask = gt_mask.cuda(non_blocking=True)

        output = model(gt, gt_mask)

        losses = criterion(output, gt, gt_mask, args, epoch)
        loss, kl_loss, l2_loss, topdown_kl, beta = losses['loss'], losses['kl'], losses['l2'], losses['topdown_kl'], losses['beta']
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
        avg_meters['kl_avg_meter'].update(kl_loss.detach().item(), data['set'].size(0))
        avg_meters['l2_avg_meter'].update(l2_loss.detach().item(), data['set'].size(0))
        avg_meters['totalloss_avg_meter'].update(loss.detach().item(), data['set'].size(0))

        if step % args.log_freq == 0:
            duration = time.time() - start_time
            start_time = time.time()

            print("[Rank %d] Epoch %d Batch [%2d/%2d] Time [%3.2fs] Loss %2.5f KL %2.5f L2 %2.5f"
                  % (args.local_rank, epoch, bidx, len(train_loader), duration,
                     loss.detach().item(), kl_loss.detach().item(), l2_loss.detach().item()))

            if logger is not None:
                logger.add_scalar('train kl loss', kl_loss.detach().item(), step)
                logger.add_scalar('train l2 loss', l2_loss.detach().item(), step)
                logger.add_scalar('train total loss', loss.detach().item(), step)
                logger.add_scalar('grad_norm', total_norm, step)

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot([kl_per_dim.detach().item() for kl_per_dim in topdown_kl])
                logger.add_figure('train top-down kl', fig, step)
                plt.close(fig)
        # assert after logging and optimizing to sync subprocesses
        kl_finite = math.isfinite(kl_loss.detach().item())
        l2_finite = math.isfinite(l2_loss.detach().item())
        loss_finite = math.isfinite(loss.detach().item())

        assert kl_finite
        assert l2_finite
        assert loss_finite

    if logger is not None:
        logger.add_scalar('train kl loss (epoch)', avg_meters['kl_avg_meter'].avg, epoch)
        logger.add_scalar('train l2 loss (epoch)', avg_meters['l2_avg_meter'].avg, epoch)
        logger.add_scalar('train total loss (epoch)', avg_meters['totalloss_avg_meter'].avg, epoch)
        logger.add_scalar('beta (epoch)', beta, epoch)
        avg_meters['kl_avg_meter'].reset()
        avg_meters['l2_avg_meter'].reset()
        avg_meters['totalloss_avg_meter'].reset()

def train_one_epoch_supervised(epoch, model, criterion, args, train_loader, avg_meters, logger):
    start_time = time.time()
    model.train()
    for bidx, data in enumerate(train_loader):
        step = bidx + len(train_loader) * epoch
        bsize = data['set'].size(0)
        output = model(data['set'].cuda(), data['set_mask'].cuda())
        preds = output['predictions']
        # prepare categorical targets for cross entropy loss
        target_classes, targets = np.unique(data['mid'], return_inverse=True)
        loss = criterion(preds, torch.tensor(targets).squeeze().cuda())
        acc = torch.sum(torch.argmax(preds, dim=1) ==  torch.tensor(targets).squeeze().cuda())/len(torch.argmax(preds, dim=1) ==  torch.tensor(targets).squeeze().cuda())
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

def validate(model, args, val_loader, epoch, criterion, logger, save_dir):
    model.eval()
    with torch.no_grad():
        #val_res = validate_reconstruct(val_loader, model, args, args.max_validate_shapes, save_dir) #orig code used this line; broke on rnaseq data due to unmask() function
        val_res = validate_reconstruct_l2(epoch, val_loader, model, criterion, args, logger)
        for k, v in val_res.items():
            if not isinstance(v, float):
                v = v.cpu().detach().item()
            if logger is not None and v is not None:
                logger.add_scalar(f'val_reconstruct/{k}', v, epoch)
        if not args.val_recon_only:
            val_sample_res = validate_sample(val_loader, model, args, args.max_validate_shapes, save_dir)
            for k, v in val_sample_res.items():
                if not isinstance(v, float):
                    v = v.cpu().detach().item()
                if logger is not None and v is not None:
                    logger.add_scalar(f'val_sample/{k}', v, epoch)
            val_res.update(val_sample_res)
    return val_res

def validate_supervised(model, args, val_loader, epoch, criterion, logger):
    model.eval()
    start_time = time.time()
    acc_meter = AverageValueMeter()
    loss_meter = AverageValueMeter()
    for bidx, data in enumerate(val_loader):
        bsize = data['set'].size(0)
        output = model(data['set'].cuda(), data['set_mask'].cuda())
        preds = output['predictions']
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

def visualize(model, args, val_loader, epoch, logger):
    model.eval()
    with torch.no_grad():
        visualize_reconstruct(val_loader, model, args, logger, epoch)
        visualize_sample(val_loader, model, args, logger, epoch)
        visualize_interpolate(val_loader, model, args, logger, epoch)
        visualize_mix(val_loader, model, args, logger, epoch)
