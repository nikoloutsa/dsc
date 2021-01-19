import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler


from data import get_datasets
from models import get_model
from utils.device import configure_device
from utils.optimizers import get_optimizer
from utils.callbacks import TimingCallback
from utils.helpers import *

EPOCHS = 60
BASE_LEARNING_RATE = 0.01
BS_PER_GPU = 64
LR_SCHEDULE = [(0.1, 30), (0.01, 45)]

def schedule(epoch):
    #return lr * (0.1 ** (epoch // 30 ))

    initial_learning_rate = BASE_LEARNING_RATE * BS_PER_GPU / 128
    learning_rate = initial_learning_rate
    for mult, start_epoch in LR_SCHEDULE:
      if epoch >= start_epoch:
        learning_rate = initial_learning_rate * mult
      else:
        break
    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate

def main():
    # init
    args = parse_args()
    ##rank, n_ranks = init_workers(args.distributed)

    # load configuration
    config = load_config(args.config)
    train_config = config['training']
    output_dir = os.path.expandvars(config['output_dir'])
    checkpoint_format = os.path.join(output_dir, 'checkpoints','checkpoint-{epoch}.h5')

    os.makedirs(output_dir, exist_ok=True)

    # logging
    config_logging(verbose=args.verbose, output_dir=output_dir)
    logging.debug('Configuration: %s',config)

    # Load Data
    train_dataset, test_dataset = get_datasets(batch_size=train_config['batch_size'],
                                        **config['data'])
    # Build Model
    model = get_model(**config['model'])

    # Configure Optimizer
    opt = get_optimizer(distributed=args.distributed, **config['optimizer'])

    model.compile(
              loss=train_config['loss'],
              optimizer=opt,
              metrics=train_config['metrics']
              )

    #model.summary()

    callbacks = []

    os.makedirs(os.path.dirname(checkpoint_format), exist_ok=True)
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_format))

    # Timing
    timing_callback = TimingCallback()
    callbacks.append(timing_callback)

    # Adjust Learning Rate
    lr = config['optimizer']['lr']
    lr_schedule_callback = LearningRateScheduler(schedule)
    #callbacks.append(lr_schedule_callback)
    
    steps_per_epoch = len(train_dataset) 
    
    hist = model.fit(train_dataset,
                    epochs=train_config['n_epochs'],
                    steps_per_epoch=steps_per_epoch,
                    validation_data=test_dataset,
                    validation_steps=len(test_dataset),
                    callbacks=callbacks
                    )

    # Print some best-found metrics
    if 'val_acc' in hist.history.keys():
        print('Best validation accuracy: %.3f',
                    max(hist.history['val_acc']))
    if 'val_top_k_categorical_accuracy' in hist.history.keys():
        print('Best top-5 validation accuracy: %.3f',
                    max(hist.history['val_top_k_categorical_accuracy']))
    print('Average time per epoch: %.3f s',
                np.mean(timing_callback.times))

if __name__ == '__main__':
    main()

