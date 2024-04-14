#!/usr/bin/env python

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.multiprocessing as mp

import utils
import models.builer as builder
import models.mrl as MRL
import dataloader

def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('--arch', default='vgg16', type=str, 
                        help='backbone architechture')
    parser.add_argument('--train_list', type=str)
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', 
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', 
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--mrl', '--matryoshka', default=False, type=bool,
                        help='Use matryoshka layer and loss')

    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('--pth-save-fold', default='results/tmp', type=str,
                        help='The folder to save pths')
    parser.add_argument('--pth-save-epoch', default=1, type=int,
                        help='The epoch to save pth')
    parser.add_argument('--parallel', type=int, default=1, 
                        help='1 for parallel, 0 for non-parallel')
    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')                                            

    args = parser.parse_args()

    return args

def main(args):
    print('=> torch version : {}'.format(torch.__version__))
    ngpus_per_node = torch.cuda.device_count()
    print('=> ngpus : {}'.format(ngpus_per_node))

    if args.parallel == 1: 
        # single machine multi card       
        args.gpus = ngpus_per_node
        args.nodes = 1
        args.nr = 0
        args.world_size = args.gpus * args.nodes

        args.workers = int(args.workers / args.world_size)
        args.batch_size = int(args.batch_size / args.world_size)
        mp.spawn(main_worker, nprocs=args.gpus, args=(args,))
    else:
        args.world_size = 1
        main_worker(ngpus_per_node, args)
    
def main_worker(gpu, args):
    utils.init_seeds(1 + gpu, cuda_deterministic=False)
    if args.parallel == 1:
        args.gpu = gpu
        args.rank = args.nr * args.gpus + args.gpu

        torch.cuda.set_device(gpu)
        torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)  
           
    else:
        # two dummy variable, not real
        args.rank = 0
        args.gpus = 1 
    if args.rank == 0:
        print('=> modeling the network {} ...'.format(args.arch))
    model = builder.BuildAutoEncoder(args)
    
    # Load ckpt if given
    if args.resume:
        print('=> loading pth from {} ...'.format(args.resume))
        model = utils.load_dict(args.resume, model)
    
    if args.rank == 0:       
        total_params = sum(p.numel() for p in model.parameters())
        print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))
    
    if args.rank == 0:
        print('=> building the oprimizer ...')
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr,)
    optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)    
    if args.rank == 0:
        print('=> building the dataloader ...')
    train_loader = dataloader.train_loader(args)

    if args.rank == 0:
        print('=> building the criterion ...')
    
    # if args.mrl:
    #     print("Using Matryoshka Loss ...")
    #     criterion = MRL.Matryoshka_MSE_Loss()
    # else:
    criterion = nn.MSELoss()

    global iters
    iters = 0

    model.train()
    if args.rank == 0:
        print('=> starting training engine ...')
    for epoch in range(args.start_epoch, args.epochs):
        
        global current_lr
        current_lr = utils.adjust_learning_rate_cosine(optimizer, epoch, args)

        train_loader.sampler.set_epoch(epoch)
        
        # print("Epoch 1")
        # train for one epoch
        do_train(train_loader, model, criterion, optimizer, epoch, args)

        # save pth
        if epoch % args.pth_save_epoch == 0 and args.rank == 0:
            state_dict = model.state_dict()

            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': state_dict,
                    'optimizer' : optimizer.state_dict(),
                },
                os.path.join(args.pth_save_fold, '{}.pth'.format(str(epoch).zfill(3)))
            )
            
            print(' : save pth for epoch {}'.format(epoch + 1))


def do_train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = utils.AverageMeter('Time', ':6.2f')
    data_time = utils.AverageMeter('Data', ':2.2f')
    losses = utils.AverageMeter('Loss', ':.4f')
    learning_rate = utils.AverageMeter('LR', ':.4f')
    
    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, learning_rate],
        prefix="Epoch: [{}]".format(epoch+1))
    end = time.time()

    # update lr
    learning_rate.update(current_lr)

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        global iters
        iters += 1
         
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()
        # output = model(input, mrl_dims=[32, 64, 128, 256, 512]) 

        output = model(input)
        # compute gradient and do solver step
        
        if len(output) > 1:
            loss_ = []
            # backward
            for j in range(len(output)):
                loss_.append(criterion(output[j], target))
            
            print(loss_)
            loss = torch.sum(torch.stack(loss_))
                
        else:
            loss = criterion(output[0], target)

        loss.backward()
        # print("Loss: ", loss.item())

        # update weights
        optimizer.step()
        # print("Optimized")
        # print(i, args.print_freq, args.rank)
        # syn for logging
        # torch.cuda.synchronize()

        # record loss
        # print("here")
        losses.update(loss.item(), input.size(0))          

        # measure elapsed time
        if args.rank == 0:
            batch_time.update(time.time() - end)        
            end = time.time()   

        # print(i, args.print_freq, args.rank)
        if i % args.print_freq == 0 and args.rank == 0:
            progress.display(i)

if __name__ == '__main__':

    args = get_args()

    main(args)


