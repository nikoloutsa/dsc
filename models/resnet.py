import tensorflow as tf
from tensorflow import keras

def ResNet50(input_shape=(32,32,3), n_classes=10):
    model = keras.applications.ResNet50(input_shape=input_shape,classes=n_classes,weights=None)
    return model
