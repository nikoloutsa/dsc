import os
import sys
import json

import tensorflow as tf
import numpy as np

#disable all gpus
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()  

def mnist_dataset(batch_size):
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data(path='/users/staff/nikoloutsa/projects/dsc/data/mnist.npz')
  # The `x` arrays are in uint8 and have values in the range [0, 255].
  # You need to convert them to float32 with values in the range [0, 1]
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
  return train_dataset

def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.Input(shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model

## single worker
#batch_size = 64
#single_worker_dataset = mnist_dataset(batch_size)
#single_worker_model = build_and_compile_cnn_model()
#single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)

## multiple worker
#options = tf.data.Options()
## not recommended, each replica process every example
#options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

per_worker_batch_size = 64
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

print(num_workers,tf_config)

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_dataset(global_batch_size)

with strategy.scope():
      # Model building/compiling need to be within `strategy.scope()`.
        multi_worker_model = build_and_compile_cnn_model()


multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)

# Currently model.predict doesn't work with MultiWorkerMirroredStrategy.
