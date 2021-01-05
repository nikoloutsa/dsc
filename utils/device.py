import tensorflow as tf

def configure_device(gpu_rank=None):
    if gpu_rank is not None:
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        gpus = tf.config.experimental.list_physical_devices('GPU')

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[gpu_rank], 'GPU')
