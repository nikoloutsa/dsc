import argparse
import os
import numpy as np
import json
from timeit import default_timer as timer

import tensorflow as tf
from tensorflow.keras import applications

# Benchmark settings
parser = argparse.ArgumentParser(description='TensorFlow Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='ResNet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda
device = 'GPU' if args.cuda else 'CPU'
#strategy = tf.distribute.MirroredStrategy()
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])
world_size = strategy.num_replicas_in_sync

verbose = 1 if tf_config['task']['index'] == 0 else 0 

if verbose:
    print('Model: %s' % args.model)
    print('Batch size: %d' % args.batch_size)
    print('Number of %ss: %d' % (device,world_size))



# Set up standard model.
opt = tf.optimizers.SGD(0.01)

# Synthetic dataset
data = tf.random.uniform([args.batch_size, 224, 224, 3])
target = tf.random.uniform([args.batch_size, 1], minval=0, maxval=999, dtype=tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((data, target)).cache().repeat().batch(args.batch_size)


with strategy.scope():
    model = getattr(applications, args.model)(weights=None)
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                  optimizer=opt)

callbacks = []

class TimingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.img_secs = []
        self.time_secs = []

    def on_train_end(self, logs=None):
        img_sec_mean = np.mean(self.img_secs)
        img_sec_conf = 1.96 * np.std(self.img_secs)
        if verbose:
            print('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
            print('Total img/sec on %d %s(s): %.1f +-%.1f' %
                 (world_size, device, world_size * img_sec_mean, world_size * img_sec_conf))

    def on_epoch_begin(self, epoch, logs=None):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs=None):
        time = timer() - self.starttime
        img_sec = args.batch_size * args.num_batches_per_iter / time
        time_sec = time
        if verbose:
            print('Iter #%d: %.1f img/sec per %s' % (epoch, img_sec, device))
        # skip warm up epoch
        if epoch > 0:
            self.img_secs.append(img_sec)
            self.time_secs.append(time_sec)

timing = TimingCallback()
callbacks.append(timing)

# Train the model.
model.fit(
    dataset,
    batch_size=args.batch_size,
    steps_per_epoch=args.num_batches_per_iter,
    callbacks=callbacks,
    epochs=args.num_iters,
    verbose=0,
)

if verbose:
    print('Steps per epoch: {}'.format(args.num_batches_per_iter))
    print('Average time per epoch: {:.3f} s'.format(np.mean(timing.time_secs)))
    print('Total time: {:.3f} s'.format(world_size * np.mean(timing.time_secs)))


