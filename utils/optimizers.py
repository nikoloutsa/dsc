
import tensorflow as tf
from tensorflow import keras

def get_optimizer(name, lr, lr_scaling='linear', **opt_args):
    # Horovod: adjust learning rate based on number of GPUs.
    if lr_scaling == 'linear':
        lr = lr * 1.0

    # Construct the optimizer
    OptType = getattr(keras.optimizers, name)
    opt = OptType(lr=lr, **opt_args)

    return opt
