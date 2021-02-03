import torch
import torch.nn as nn
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import horovod.torch as hvd
import os
import math
from tqdm import tqdm

import time
import numpy as np
from utils.helpers import load_config

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--wd', '--weight-decay', default=0.00005, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-c','--config', default='', type=str, help='configuration file')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                            metavar='N', help='print frequency (default: 10)')

def main():
    args = parser.parse_args()
    # load configuration
    config = load_config(args.config)
    output_dir = os.path.expandvars(config['output_dir'])
    args.checkpoint_filename = os.path.join(output_dir, 'checkpoints','checkpoint.pth.tar')

    os.makedirs(os.path.dirname(args.checkpoint_filename), exist_ok=True)
    #os.makedirs(output_dir, exist_ok=True)

    # set args from config file
    args.data = config['data']['path']
    #'mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel'
    args.epochs = config['training']['n_epochs']
    args.batch_size = config['training']['batch_size'] 
    args.loss = config['training']['loss']
    args.lr = config['optimizer']['lr']
    args.momentum = config['optimizer']['momentum']
    args.optimizer = config['optimizer']['name']
    args.arch = config['model']['name']
    args.workers = 4
    args.gpu = None


    ngpus_per_node = torch.cuda.device_count()
    print("Number of devices per node: {}".format(ngpus_per_node))

    hvd.init()
    print("rank",hvd.rank())

    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())

    cudnn.benchmark = True

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    kwargs = {'num_workers': 4, 'pin_memory': True}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = \
        datasets.ImageFolder(traindir,
                             transform=transforms.Compose([
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))
    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, **kwargs)

    val_dataset = \
        datasets.ImageFolder(valdir,
                             transform=transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize, 
                             ]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False, **kwargs)


    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    # Move model to GPU.
    model.cuda()

    # define loss function (criterion) and optimizer
    criType = getattr(nn, args.loss)
    criterion = criType().cuda()

    # Horovod: scale learning rate by the number of GPUs.
    OptType = getattr(torch.optim, args.optimizer)
    optimizer = OptType(model.parameters(), 
                        lr=(args.base_lr * hvd.size()), 
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression,
        op=hvd.Average)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    times = []
    for epoch in range(0, args.epochs):
        starttime = time.time()

        train(train_loader, train_sampler, model, criterion, optimizer, epoch, verbose, args)
        validate(val_loader, model, criterion, epoch, verbose, args)
        #save_checkpoint(epoch)

        epoch_time = time.time() - starttime
        print('Epoch {}/{} - {:.3f}s'.format(epoch,args.epochs,epoch_time))
        times.append(epoch_time)

    print('Steps per epoch: {}'.format(len(train_loader)))
    print('Validation steps per epoch: {}'.format(len(val_loader)))
    print('Average time per epoch: {:.3f} s'.format(
        np.mean(times)))

def train(train_loader, train_sampler, model, criterion, optimizer, epoch, verbose, args):
    batch_time = Metric('Time')
    data_time = Metric('Data')
    losses = Metric('Loss')
    top1 = Metric('Acc@1')

    # switch to train mode
    model.train()
    train_sampler.set_epoch(epoch)

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        end = time.time()
        for batch_idx, (images, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update_time(time.time() - end)

            adjust_learning_rate(train_loader, optimizer, epoch, batch_idx, args)

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            top1.update(accuracy(output, target))
            losses.update(loss)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            """
            # Split data into sub-batches of size batch_size
            for i in range(0, len(images), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)
                top1.update(accuracy(output, target_batch))
                loss = F.cross_entropy(output, target_batch)
                losses.update(loss)
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(images)) / args.batch_size))
                loss.backward()
            """
            # Gradient is applied across all ranks
            optimizer.step()

            batch_time.update_time(time.time() - end)
            end = time.time()
            t.set_postfix({'Loss': losses.avg.item(),
                           'Acc@1': 100. * top1.avg.item(),
                           'Data': data_time.avg.item(),
                           'Time': batch_time.avg.item()})
            t.update(1)

def validate(val_loader, model, criterion, epoch, verbose, args):
    model.eval()
    losses = Metric('Loss')
    top1 = Metric('Acc@1')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for batch_idx, (images, target) in enumerate(val_loader):
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                output = model(images)

                loss = criterion(output, target)
                losses.update(loss)
                top1.update(accuracy(output, target))
                t.set_postfix({'Loss': losses.avg.item(),
                               'Acc@1': 100. * top1.avg.item()})
                t.update(1)


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(train_loader, optimizer, epoch, batch_idx, args):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch, args):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    def update_time(self, val):
        self.sum += val
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
