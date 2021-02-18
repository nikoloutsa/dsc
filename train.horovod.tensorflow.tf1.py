#
# ResNet-50 model training using Keras and Horovod.
#
# This model is an example of a computation-intensive model that achieves good accuracy on an image
# classification task.  It brings together distributed training concepts such as learning rate
# schedule adjustments with a warmup, randomized data reading, and checkpointing on the first worker
# only.
#
# Note: This model uses Keras native ImageDataGenerator and not the sophisticated preprocessing
# pipeline that is typically used to train state-of-the-art ResNet-50 model.  This results in ~0.5%
# increase in the top-1 validation error compared to the single-crop top-1 validation error from
# https://github.com/KaimingHe/deep-residual-networks.
#
from __future__ import print_function

import argparse
import keras
from keras import backend as K
from keras.preprocessing import image
import tensorflow as tf
import horovod.keras as hvd
import os
import time
import numpy as np
import time

from utils.callbacks_tf1 import TimingCallback
from utils.helpers import *

start_time = time.time()

# initialization
args = parse_args()

# load configuration
config = load_config(args.config)

# load configuration
config = load_config(args.config)
train_config = config['training']
args.batch_size = train_config['batch_size']
args.val_batch_size = args.batch_size
args.base_lr = config['optimizer']['lr']
args.epochs = train_config['n_epochs']
args.warmup_epochs =  train_config['lr_warmup_epochs']
args.wd = 0.00005
args.momentum = 0.9
args.train_dir = config['data']['path']+"/train"
args.val_dir = config['data']['path']+"/val"
output_dir = os.path.expandvars(config['output_dir'])
args.checkpoint_format = os.path.join(output_dir, 'checkpoints','checkpoint-{epoch}.h5')
args.fp16_allreduce = False

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0
#for try_epoch in range(args.epochs, 0, -1):
#    if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
#        resume_from_epoch = try_epoch
#        break

# Horovod: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
resume_from_epoch = hvd.broadcast(resume_from_epoch, 0, name='resume_from_epoch')

# Horovod: print logs on the first worker.
verbose = 1 if hvd.rank() == 0 else 0

# Training data iterator.
train_gen = image.ImageDataGenerator(
    width_shift_range=0.33, height_shift_range=0.33, zoom_range=0.5, horizontal_flip=True,
    preprocessing_function=keras.applications.resnet50.preprocess_input)
train_iter = train_gen.flow_from_directory(args.train_dir,
                                           batch_size=args.batch_size,
                                           target_size=(224, 224))

# Validation data iterator.
test_gen = image.ImageDataGenerator(
    zoom_range=(0.875, 0.875), preprocessing_function=keras.applications.resnet50.preprocess_input)
test_iter = test_gen.flow_from_directory(args.val_dir,
                                         batch_size=args.val_batch_size,
                                         target_size=(224, 224))

# Set up standard ResNet-50 model.
model = keras.applications.resnet50.ResNet50(weights=None)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast both model and optimizer weights
# to other workers.
if resume_from_epoch > 0 and hvd.rank() == 0:
    model = hvd.load_model(args.checkpoint_format.format(epoch=resume_from_epoch),
                           compression=compression)
else:
    # ResNet-50 model that is included with Keras is optimized for inference.
    # Add L2 weight decay & adjust BN settings.
    model_config = model.get_config()
    for layer, layer_config in zip(model.layers, model_config['layers']):
        if hasattr(layer, 'kernel_regularizer'):
            regularizer = keras.regularizers.l2(args.wd)
            layer_config['config']['kernel_regularizer'] = \
                {'class_name': regularizer.__class__.__name__,
                 'config': regularizer.get_config()}
        if type(layer) == keras.layers.BatchNormalization:
            layer_config['config']['momentum'] = 0.9
            layer_config['config']['epsilon'] = 1e-5

    model = keras.models.Model.from_config(model_config)

    # Horovod: adjust learning rate based on number of GPUs.
    opt = keras.optimizers.SGD(lr=args.base_lr * hvd.size(),
                               momentum=args.momentum)

    # Horovod: add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt, compression=compression)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy', 'top_k_categorical_accuracy'])

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard, or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=args.warmup_epochs, verbose=verbose),

    # Horovod: after the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=args.warmup_epochs, end_epoch=30, multiplier=1.),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=30, end_epoch=60, multiplier=1e-1),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=60, end_epoch=80, multiplier=1e-2),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=80, multiplier=1e-3),
]

# Timing
timing_callback = TimingCallback()
callbacks.append(timing_callback)

# Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
if hvd.rank() == 0:
    pass
    #callbacks.append(keras.callbacks.ModelCheckpoint(args.checkpoint_format))
    #callbacks.append(keras.callbacks.TensorBoard(args.log_dir))

# Train the model. The training will randomly sample 1 / N batches of training data and
# 3 / N batches of validation data on every worker, where N is the number of workers.
# Over-sampling of validation data helps to increase probability that every validation
# example will be evaluated.
#steps_per_epoch=len(train_iter) // hvd.size()
#validation_steps=3 * len(test_iter) // hvd.size())

steps_per_epoch=32
validation_steps=32
print("Steps per epoch:",steps_per_epoch)

hist = model.fit_generator(train_iter,
                    steps_per_epoch=steps_per_epoch,
                    callbacks=callbacks,
                    epochs=args.epochs,
                    verbose=verbose,
                    workers=1,
                    initial_epoch=resume_from_epoch,
                    validation_data=test_iter,
                    validation_steps=validation_steps)

# Print some best-found metrics
print('Steps per epoch: {}'.format(steps_per_epoch))
print('Validation steps per epoch: {}'.format(validation_steps))
if 'val_accuracy' in hist.history.keys():
    print('Best validation accuracy: {:.3f}'.format(
        max(hist.history['val_accuracy'])))
if 'val_top_k_categorical_accuracy' in hist.history.keys():
    print('Best top-5 validation accuracy: {:.3f}'.format(
        max(hist.history['val_top_k_categorical_accuracy'])))
print('Average time per epoch: {:.3f} s'.format(
    np.mean(timing_callback.times)))

print("--- %s seconds ---" % (time.time() - start_time))
