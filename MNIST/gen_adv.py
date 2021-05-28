from tensorflow import keras
import tensorflow as tf
import numpy as np
from attack import FGSM, PGD
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 



def load_mnist(path="./mnist.npz"):
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


path = "./mnist.npz"
x_train, x_test, y_train, y_test = load_mnist(path)


# load your model 
model = keras.models.load_model("./Lenet5_mnist.h5")

fgsm = FGSM(model, ep=0.3, isRand=True)
pgd = PGD(model, ep=0.3, epochs=10, isRand=True)

# generate adversarial examples at once. 
advs, labels, fols, ginis = fgsm.generate(x_train, y_train)
np.savez('./FGSM_TrainFull.npz', advs=advs, labels=labels, fols=fols, ginis=ginis)

advs, labels, fols, ginis = pgd.generate(x_train, y_train)
np.savez('./PGD_TrainFull.npz', advs=advs, labels=labels, fols=fols, ginis=ginis)

advs, labels, _, _ = fgsm.generate(x_test, y_test)
np.savez('./FGSM_Test.npz', advs=advs, labels=labels)

advs, labels, _, _ = pgd.generate(x_test, y_test)
np.savez('./PGD_Test.npz', advs=advs, labels=labels)