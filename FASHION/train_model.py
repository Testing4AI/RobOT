from tensorflow import keras
import tensorflow as tf
import numpy as np
from models import Lenet5, ConvNet_1

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        

def load_fashion(path="./fashion.npz"):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return x_train, x_test, y_train, y_test


path = "./fashion.npz"
x_train, x_test, y_train, y_test = load_fashion(path)


lenet5 = Lenet5()
lenet5.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
lenet5.fit(x_train, y_train, epochs=10, batch_size=64)

lenet5.evaluate(x_test, y_test)

lenet5.save("./Lenet5_fashion.h5")
