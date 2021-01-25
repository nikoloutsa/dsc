
import argparse
import yaml
import logging
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    #add_arg('config', nargs='?')
    add_arg('-c','--config', default='', type=str, help='configuration file')
    add_arg('-d', '--distributed', action='store_true')
    add_arg('-v','--verbose', action='store_true')

    add_arg('-j', '--workers', default=1, type=int, metavar='N',
            help='number of data loading workers (default: 1)')
    add_arg('--start-epoch', default=0, type=int, metavar='N',
            help='manual epoch number (useful on restarts)')
    add_arg('--wd', '--weight-decay', default=1e-4, type=float,
            metavar='W', help='weight decay (default: 1e-4)',
            dest='weight_decay')
    add_arg('-p', '--print-freq', default=10, type=int,
            metavar='N', help='print frequency (default: 10)')
    add_arg('--resume', default='', type=str, metavar='PATH',
            help='path to latest checkpoint (default: none)')
    add_arg('--pretrained', dest='pretrained', action='store_true',
            help='use pre-trained model')
    add_arg('--world-size', default=-1, type=int,
            help='number of nodes for distributed training')
    add_arg('--rank', default=-1, type=int,
            help='node rank for distributed training')
    add_arg('--dist-url', default='tcp://224.66.41.62:23456', type=str,
            help='url used to set up distributed training')
    add_arg('--dist-backend', default='nccl', type=str,
            help='distributed backend')
    add_arg('--gpu', default=None, type=int,
            help='GPU id to use.')
    add_arg('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    return parser.parse_args()

def config_logging(verbose, output_dir):
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(log_level)
    file_handler = logging.FileHandler(os.path.join(output_dir, 'out.log'), mode='w')
    file_handler.setLevel(log_level)
    logging.basicConfig(level=log_level, format=log_format,
                        handlers=[stream_handler, file_handler])

def load_config(config_file):
    with open(config_file) as f:
        config = yaml.safe_load(f)
    return config
