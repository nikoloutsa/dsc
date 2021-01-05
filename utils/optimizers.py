
import tensorflow as tf
from tensorflow import keras
import horovod.tensorflow.keras as hvd
   
def get_optimizer(name, lr, lr_scaling='linear', n_ranks=1, distributed=False, **opt_args):
    print(opt_args)
    print(lr)
    # Horovod: adjust learning rate based on number of GPUs.
    if lr_scaling == 'linear':
        lr = lr * n_ranks

    # Construct the optimizer
    OptType = getattr(keras.optimizers, name)
    opt = OptType(lr=lr, **opt_args)

    if distributed:
        opt = hvd.DistributedOptimizer(opt)
      
    return opt
