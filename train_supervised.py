import os
import random
import datetime
from pathlib import Path
from copy import deepcopy

import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing
import deepspeed

from args import get_args
from models.networks import SetVAE, SetPredictor
from utils import AverageValueMeter, set_random_seed, save, resume, validate_reconstruct_l2
from datasets import get_datasets
from engine import train_one_epoch, train_one_epoch_supervised, validate, validate_supervised, visualize

# torch.autograd.set_detect_anomaly(True)

def main_worker(save_dir, args, cur_time):
    # basic setup
    cudnn.benchmark = True

    if args.model_name is not None:
        log_dir = Path(save_dir) / "tensorboard_logs/{}".format(args.model_name)
        save_dir = Path(save_dir) / "checkpoints/{}".format(args.model_name)
        if not Path(save_dir).exists():
            Path(save_dir).mkdir(exist_ok=True, parents=True)
    else:
        log_dir = Path(save_dir) / f"tensorboard_logs/{cur_time}"
        save_dir = Path(save_dir) / f"checkpoints/{cur_time}"
        if not Path(save_dir).exists():
            Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    if args.local_rank == 0:
        logger = SummaryWriter(log_dir)
    else:
        logger = None

    deepspeed.init_distributed(dist_backend='nccl')
    torch.cuda.set_device(args.local_rank)

    # initialize datasets and loaders
    train_dataset, val_dataset, train_loader, val_loader, num_groups = get_datasets(args)
    args.num_targets = num_groups
    model = SetPredictor(args)
    print("Model initialized")
    parameters = model.parameters()

    n_parameters = sum(p.numel() for p in parameters if p.requires_grad)
    print(f'number of params: {n_parameters}')
    try:
        n_gen_parameters = sum(p.numel() for p in model.init_set.parameters() if p.requires_grad) + \
                           sum(p.numel() for p in model.pre_decoder.parameters() if p.requires_grad) + \
                           sum(p.numel() for p in model.decoder.parameters() if p.requires_grad) + \
                           sum(p.numel() for p in model.post_decoder.parameters() if p.requires_grad) + \
                           sum(p.numel() for p in model.output.parameters() if p.requires_grad)
        print(f'number of generator params: {n_gen_parameters}')
    except AttributeError:
        pass

    optimizer, loss = model.make_optimizer(args)



    # initialize the learning rate scheduler
    if args.scheduler == 'exponential':
        assert not (args.warmup_epochs > 0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.exp_decay)
    elif args.scheduler == 'step':
        assert not (args.warmup_epochs > 0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 2, gamma=0.1)
    elif args.scheduler == 'linear':
        def lambda_rule(ep):
            lr_w = min(1., ep / args.warmup_epochs) if (args.warmup_epochs > 0) else 1.
            lr_l = 1.0 - max(0, ep - 0.5 * args.epochs) / float(0.5 * args.epochs)
            return lr_l * lr_w

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.scheduler == 'cosine':
        assert not (args.warmup_epochs > 0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        # Fake SCHEDULER
        def lambda_rule(ep):
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    # extract collate_fn
    if args.distributed:
        torch.multiprocessing.set_sharing_strategy('file_system') #try to fix error "OSError: [Errno 24] Too many open files"
        collate_fn = deepcopy(train_loader.collate_fn)
        model, optimizer, train_loader, scheduler = deepspeed.initialize( #RB what they call "model" here is really "engine" from deepspeed, that's why it has a .optimizer etc that they can call later
            args=args,
            model=model,
            optimizer=optimizer,
            model_parameters=parameters,
            training_data=train_dataset,
            collate_fn=collate_fn,
            lr_scheduler=scheduler
        )
    print("Deepspeed initialized")

    # resume checkpoints
    start_epoch = 0
    if args.resume_checkpoint is None and Path(Path(save_dir) / 'checkpoint-latest.pt').exists():
        args.resume_checkpoint = os.path.join(save_dir, 'checkpoint-latest.pt')  # use the latest checkpoint
        print('Resumed from: ' + args.resume_checkpoint)
    if args.resume_checkpoint is not None:
        if args.distributed:
            if args.resume_optimizer:
                model.module, model.optimizer, model.lr_scheduler, start_epoch = resume(
                    args.resume_checkpoint, model.module, model.optimizer, scheduler=model.lr_scheduler,
                    strict=(not args.resume_non_strict))
            else:
                model.module, _, _, start_epoch = resume(
                    args.resume_checkpoint, model.module, optimizer=None, strict=(not args.resume_non_strict))
        else:
            if args.resume_optimizer:
                model, optimizer, scheduler, start_epoch = resume(
                    args.resume_checkpoint, model, optimizer, scheduler=scheduler, strict=(not args.resume_non_strict))
            else:
                model, _, _, start_epoch = resume(
                    args.resume_checkpoint, model, optimizer=None, strict=(not args.resume_non_strict))

    # save dataset statistics
    if args.local_rank == 0:
        train_dataset.save_statistics(save_dir)
        val_dataset.save_statistics(save_dir)

    # main training loop
    avg_meters = {
        'loss_avg_meter': AverageValueMeter(),
        'acc_avg_meter': AverageValueMeter(),
    }

    assert args.distributed

    epoch = start_epoch
    if epoch == 0: #if training a new model
        best_val_totalloss = 1e30
    else: #if reading in a saved model
        print("NOT YET IMPLEMENTED")
        # TO DO
        #best_val_totalloss #should be calculated based on best saved model
    print("Start epoch: %d End epoch: %d" % (start_epoch, args.epochs))
    for epoch in range(start_epoch, args.epochs):
        # train for one epoch
        train_one_epoch_supervised(epoch, model, loss, args, train_loader, avg_meters, logger)

        # evaluate on the validation set
        if args.local_rank == 0:
            if epoch % args.val_freq == 0 and epoch != 0:
                model.eval()
                with torch.no_grad():
                    val_losses = validate_supervised(model.module, args, val_loader, epoch, loss, logger)
                    for k, v in val_losses.items():
                        if not isinstance(v, float):
                            v = v.cpu().detach().item()
                        if logger is not None and v is not None:
                            logger.add_scalar(f'val_sample/{k}', v, epoch - 1)

                    if val_losses['val_loss'] < best_val_totalloss:
                        best_val_totalloss = val_losses['val_loss']
                        save(model.module, model.optimizer, model.lr_scheduler, epoch + 1,
                            Path(save_dir) / 'checkpoint-best.pt')


        # adjust the learning rate
        model.lr_scheduler.step()
        if logger is not None and args.local_rank == 0:
            logger.add_scalar('train lr', model.lr_scheduler.get_last_lr()[0], epoch)

    #run validation once after training is done
    print("running validation...")
    model.eval()
    if args.local_rank == 0:
        with torch.no_grad():
            val_losses = validate_supervised(model.module, args, val_loader, epoch, loss, logger)
            for k, v in val_losses.items():
                if not isinstance(v, float):
                    v = v.cpu().detach().item()
                if logger is not None and v is not None:
                    logger.add_scalar(f'val_sample/{k}', v, epoch)

            if val_losses['val_loss'] < best_val_totalloss:
                best_val_totalloss = val_losses['val_loss']
                save(model.module, model.optimizer, model.lr_scheduler, epoch + 1,
                    Path(save_dir) / 'checkpoint-best.pt')

            save(model.module, model.optimizer, model.lr_scheduler, epoch + 1,
                    Path(save_dir) / 'checkpoint-latest.pt')

    if logger is not None:
        logger.flush()
        logger.close()

def write_params(file, args):
    with open(file, 'a') as f:
        #write one arg per line from parser.parse_args()
        for arg in vars(args):
            f.write("{}: {}\n".format(arg, getattr(args, arg)))
    

def main():
    args = get_args()
    save_dir = Path(args.log_dir) #defaults to current directory
    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    set_random_seed(args.seed)
    cur_time = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
    if args.local_rank == 0:
        write_params(Path(save_dir) / 'param_logs' / '{}_{}_params.txt'.format(args.model_name, cur_time), args)
    main_worker(save_dir, args, cur_time)


if __name__ == '__main__':
    main()
