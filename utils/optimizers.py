
import tensorflow as tf
from tensorflow import keras
   
def get_optimizer(name, lr, **opt_args):
    # Construct the optimizer
    OptType = getattr(keras.optimizers, name)
    opt = OptType(lr=lr, **opt_args)

    return opt
