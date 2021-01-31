import tensorflow as tf
from tensorflow import keras

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3

def augmentation(x, y):
    x = tf.image.resize_with_crop_or_pad(
        x, HEIGHT + 8, WIDTH + 8)
    x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
    x = tf.image.random_flip_left_right(x)
    return x, y 

def get_datasets(batch_size, path):

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # labels to class vectors
    n_classes = 10
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_dataset = train_dataset.map(augmentation).batch(batch_size, drop_remainder=False)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=False)

    return train_dataset, test_dataset
