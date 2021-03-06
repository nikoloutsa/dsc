import numpy as np
import time
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler


from data import get_datasets
from models import get_model
from utils.device import configure_device
from utils.optimizers import get_optimizer
from utils.callbacks import TimingCallback
from utils.helpers import *

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

LR_SCHEDULE = [(0.1, 30), (0.01, 45)]

def schedule(epoch,lr):
    initial_learning_rate = lr 
    learning_rate = initial_learning_rate
    for mult, start_epoch in LR_SCHEDULE:
      if epoch >= start_epoch:
        learning_rate = initial_learning_rate * mult
      else:
        break
    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate

def main():
    # initialization
    args = parse_args()

    # load configuration
    config = load_config(args.config)
    train_config = config['training']
    output_dir = os.path.expandvars(config['output_dir'])
    checkpoint_format = os.path.join(output_dir, 'checkpoints','checkpoint-{epoch}.h5')

    os.makedirs(output_dir, exist_ok=True)

    tf_config = json.loads(os.environ['TF_CONFIG'])
    print(tf_config)
    num_workers = len(tf_config['cluster']['worker'])

    #verbose = 1 if tf_config['cluster']['task']['index'] == 0 else 0
    verbose = 1


    args.batch_size = train_config['batch_size'] * num_workers
    #args.batch_size = train_config['batch_size']
    print("multiworker batch_size:",args.batch_size)

    # logging
    config_logging(verbose=args.verbose, output_dir=output_dir)
    logging.debug('Configuration: %s',config)

    # Load Data
    train_dataset, test_dataset = get_datasets(batch_size=args.batch_size,
                                        **config['data'])
    print('Steps per epoch: {}'.format(len(train_dataset)))

    # Configure Optimizer
    lr = config['optimizer']['lr']

    lr = lr * strategy.num_replicas_in_sync
    # Construct the optimizer
    OptType = getattr(keras.optimizers, config['optimizer']['name'])
    opt = OptType(learning_rate=lr, momentum=config['optimizer']['momentum'])
    #opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

    # Compile Model
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = get_model(**config['model'])
        model.compile(
                  loss=train_config['loss'],
                  optimizer=opt,
                  metrics=train_config['metrics']
                  )

    # Print Model Summary
    #model.summary()

    # Prepare the training callbacks
    callbacks = []

    os.makedirs(os.path.dirname(checkpoint_format), exist_ok=True)
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_format))

    # Timing
    timing_callback = TimingCallback()
    callbacks.append(timing_callback)

    # Adjust Learning Rate
    lr_schedule_callback = LearningRateScheduler(schedule)
    callbacks.append(lr_schedule_callback)
   
    # Train the model
    steps_per_epoch = len(train_dataset) 
    validation_steps = len(test_dataset) 
    #steps_per_epoch = 32 / strategy.num_replicas_in_sync
    steps_per_epoch = 128
    validation_steps = 128 

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

