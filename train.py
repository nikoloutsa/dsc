import os
import sys
import argparse
import yaml
import logging

from data import get_datasets
from models import get_model
from utils.optimizers import get_optimizer
from utils.callbacks import TimingCallback

def parse_args():
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?')
    add_arg('-v','--verbose', action='store_true')

    return parser.parse_args()

def config_logging(verbose, output_dir):
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(log_level)
    file_handler = logging.FileHandler(os.path.join(output_dir, 'out.log'), mode='w')
    file_handler.setLevel(log_level)
    logging.basicConfig(level=log_level, format=log_format,
                        handlers=[stream_handler, file_handler])

def load_config(config_file):
    with open(config_file) as f:
        config = yaml.safe_load(f)
    return config

def main():
    args = parse_args()
    config = load_config(args.config)
    train_config = config['training']
    output_dir =os.path.expandvars(config['output_dir'])

    os.makedirs(output_dir, exist_ok=True)

    config_logging(verbose=args.verbose, output_dir=output_dir)
    logging.debug('Configuration: %s',config)
  
    # Load Data
    train_gen, test_gen = get_datasets(batch_size=train_config['batch_size'],
                                        **config['data'])

    # Build Model
    model = get_model(**config['model'])
    # Configure Optimizer
    print(config['optimizer'])
    opt = get_optimizer(**config['optimizer'])

    model.compile(
                loss=train_config['loss'],
                optimizer=opt,
                metrics=train_config['metrics'])

    model.summary()

    callbacks = []

    # Timing
    timing_callback = TimingCallback()
    callbacks.append(timing_callback)

    # Train Model
    n_ranks = 1
    steps_per_epoch = len(train_gen) // n_ranks
    history = model.fit(train_gen,
                                  epochs=train_config['n_epochs'],
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=test_gen,
                                  validation_steps=len(test_gen),
                                  callbacks=callbacks,
                                  workers=1)

    # Print some best-found metrics
    if 'val_acc' in history.history.keys():
        logging.info('Best validation accuracy: %.3f',
                    max(history.history['val_acc']))
    if 'val_top_k_categorical_accuracy' in history.history.keys():
        logging.info('Best top-5 validation accuracy: %.3f',
                    max(history.history['val_top_k_categorical_accuracy']))
    logging.info('Average time per epoch: %.3f s',
                np.mean(timing_callback.times))
    np.savez(os.path.join(output_dir, 'history'),
            n_ranks=n_ranks, **history.history)

     
if __name__ == '__main__':
    main()
