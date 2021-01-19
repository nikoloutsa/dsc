import os
import sys
import argparse
import yaml
import logging
import numpy as np

import tensorflow as tf
import horovod.tensorflow.keras as hvd

from data import get_datasets
from models import get_model
from utils.device import configure_device
from utils.optimizers import get_optimizer
from utils.callbacks import TimingCallback

def parse_args():
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?')
    add_arg('-d', '--distributed', action='store_true')
    add_arg('-v','--verbose', action='store_true')

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

def init_workers(distributed=False):
    rank, n_ranks = 0, 1
    if distributed:
        hvd.init()
        rank, n_ranks = hvd.rank(), hvd.size()
    return rank, n_ranks

def load_config(config_file):
    with open(config_file) as f:
        config = yaml.safe_load(f)
    return config

def main():
    # init
    args = parse_args()
    rank, n_ranks = init_workers(args.distributed)

    # load configuration
    config = load_config(args.config)
    train_config = config['training']
    output_dir =os.path.expandvars(config['output_dir'])
    checkpoint_format = os.path.join(output_dir, 'checkpoints','checkpoint-{epoch}.h5')

    os.makedirs(output_dir, exist_ok=True)

    # logging
    config_logging(verbose=args.verbose, output_dir=output_dir)
    if rank == 0:
        logging.debug('Configuration: %s',config)

    gpu_rank = 0
    # configure devices
    if args.distributed:
        gpu_rank = hvd.local_rank()
    
    configure_device(gpu_rank=gpu_rank) 

    # Load Data
    train_gen, test_gen = get_datasets(batch_size=train_config['batch_size'],
                                        **config['data'])

    # Build Model
    model = get_model(**config['model'])

    # Configure Optimizer
    opt = get_optimizer(n_ranks=n_ranks, distributed=args.distributed, **config['optimizer'])

    # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
    # uses hvd.DistributedOptimizer() to compute gradients.
    if args.distributed:
        model.compile(
                loss=train_config['loss'],
                optimizer=opt,
                metrics=train_config['metrics'],
                experimental_run_tf_function=True
                )
    else:
        model.compile(
                loss=train_config['loss'],
                optimizer=opt,
                metrics=train_config['metrics']
                )

    if rank == 0:
        model.summary()

    callbacks = []

    if args.distributed:
        if config['optimizer']['lr_scaling'] == 'linear':
            initial_lr = config['optimizer']['lr'] * n_ranks
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        
        # Horovod: average metrics among workers at the end of every epoch.
        callbacks.append(hvd.callbacks.MetricAverageCallback())
        
        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
        warmup_epochs= train_config.get('lr_warmup_epochs', 0)
        callbacks.append(hvd.callbacks.LearningRateWarmupCallback(initial_lr=initial_lr, warmup_epochs=warmup_epochs, verbose=1))

        # Horovod: after the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
        hvd.callbacks.LearningRateScheduleCallback(initial_lr=initial_lr,
                                                   multiplier=1.,
                                                   start_epoch=warmup_epochs,
                                                   end_epoch=30),
        hvd.callbacks.LearningRateScheduleCallback(initial_lr=initial_lr, multiplier=1e-1, start_epoch=30, end_epoch=60),
        hvd.callbacks.LearningRateScheduleCallback(initial_lr=initial_lr, multiplier=1e-2, start_epoch=60, end_epoch=80),
        hvd.callbacks.LearningRateScheduleCallback(initial_lr=initial_lr, multiplier=1e-3, start_epoch=80),
   
    if rank == 0:
        os.makedirs(os.path.dirname(checkpoint_format), exist_ok=True)
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_format))

    # Timing
    timing_callback = TimingCallback()
    callbacks.append(timing_callback)

    # Horovod: write logs on worker 0.
    verbose = 1 if rank == 0 else 0

    # Train Model
    steps_per_epoch = len(train_gen) // n_ranks
    history = model.fit(train_gen,
                                  epochs=train_config['n_epochs'],
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=test_gen,
                                  validation_steps=len(test_gen),
                                  callbacks=callbacks,
                                  verbose=verbose)
    if rank == 0:
        # Print some best-found metrics
        if 'val_acc' in history.history.keys():
            logging.info('Best validation accuracy: %.3f',
                        max(history.history['val_acc']))
        if 'val_top_k_categorical_accuracy' in history.history.keys():
            logging.info('Best top-5 validation accuracy: %.3f',
                        max(history.history['val_top_k_categorical_accuracy']))
        logging.info('Average time per epoch: %.3f s',
                    np.mean(timing_callback.times))
        np.savez(os.path.join(output_dir, 'history'),
                n_ranks=n_ranks, **history.history)

     
if __name__ == '__main__':
    main()
