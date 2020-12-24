import tensorflow as tf
from tensorflow import keras

def get_datasets(batch_size):

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # labels to class vectors
    n_classes = 10
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    train_gen = keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    horizontal_flip=True)
    test_gen = keras.preprocessing.image.ImageDataGenerator()

    train_iter = train_gen.flow(x_train, y_train, batch_size=batch_size)
    test_iter = test_gen.flow(x_test, y_test, batch_size=batch_size)

    return train_iter, test_iter
