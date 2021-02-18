import numpy as np
import time

import tensorflow as tf
from tensorflow import keras
import horovod.tensorflow.keras as hvd


from data import get_datasets
from models import get_model
from utils.device import configure_device
from utils.optimizers import get_optimizer
from utils.callbacks import TimingCallback
from utils.helpers import *

def main():
    # initialization
    args = parse_args()

    # load configuration
    config = load_config(args.config)
    train_config = config['training']
    output_dir = os.path.expandvars(config['output_dir'])
    checkpoint_format = os.path.join(output_dir, 'checkpoints','checkpoint-{epoch}.h5')

    os.makedirs(output_dir, exist_ok=True)

    # logging
    config_logging(verbose=args.verbose, output_dir=output_dir)
    logging.debug('Configuration: %s',config)

    # Horovod: initialize Horovod.
    hvd.init()
    
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    # Load Data
    train_dataset, test_dataset = get_datasets(batch_size=train_config['batch_size'],
                                        **config['data'])

    # Configure Optimizer
    lr = config['optimizer']['lr']

    initial_lr = lr * hvd.size()
    # Construct the optimizer
    OptType = getattr(keras.optimizers, config['optimizer']['name'])
    opt = OptType(learning_rate=initial_lr, momentum=config['optimizer']['momentum'])
    #opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    opt = hvd.DistributedOptimizer(opt)

    model = get_model(**config['model'])
    model.compile(
              loss=train_config['loss'],
              optimizer=opt,
              metrics=train_config['metrics']
              )

    # Print Model Summary
    #model.summary()

    # Prepare the training callbacks
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        hvd.callbacks.LearningRateWarmupCallback(initial_lr=initial_lr, warmup_epochs=3, verbose=1),
        hvd.callbacks.LearningRateScheduleCallback(initial_lr=initial_lr,
                                               multiplier=1.,
                                               start_epoch=3,
                                               end_epoch=30),
        hvd.callbacks.LearningRateScheduleCallback(initial_lr=initial_lr, multiplier=1e-1, start_epoch=30, end_epoch=60),
        hvd.callbacks.LearningRateScheduleCallback(initial_lr=initial_lr, multiplier=1e-2, start_epoch=60, end_epoch=80),
        hvd.callbacks.LearningRateScheduleCallback(initial_lr=initial_lr, multiplier=1e-3, start_epoch=80),
    ]

    if hvd.rank() == 0:
        os.makedirs(os.path.dirname(checkpoint_format), exist_ok=True)
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_format))

    # Timing
    timing_callback = TimingCallback()
    callbacks.append(timing_callback)

    verbose = 1 if hvd.rank() == 0 else 0

    # Train the model
    steps_per_epoch=len(train_dataset) // hvd.size()
    validation_steps=len(test_dataset) // hvd.size()
    #validation_steps=3 * len(test_iter) // hvd.size()
    #steps_per_epoch = 32
    #validation_steps = 32
    print("Steps per epoch:",steps_per_epoch)

    hist = model.fit(train_dataset,
                    epochs=train_config['n_epochs'],
                    steps_per_epoch=steps_per_epoch,
                    validation_data=test_dataset,
                    validation_steps=validation_steps,
                    workers=1,
                    verbose=verbose,
                    callbacks=callbacks
                    )
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

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

