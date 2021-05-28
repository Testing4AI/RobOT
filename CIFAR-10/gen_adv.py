from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
from attack import FGSM, PGD
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# preprocess cifar dataset
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# load your model 
model = keras.models.load_model("./cifar10_resnet20_model.h5")

fgsm = FGSM(model, ep=0.01, isRand=True)
pgd = PGD(model, ep=0.01, epochs=10, isRand=True)

# generate adversarial examples at once. 
advs, labels, fols, ginis = fgsm.generate(x_train, y_train)
np.savez('./FGSM_TrainFull.npz', advs=advs, labels=labels, fols=fols, ginis=ginis)

advs, labels, fols, ginis = pgd.generate(x_train, y_train)
np.savez('./PGD_TrainFull.npz', advs=advs, labels=labels, fols=fols, ginis=ginis)

advs, labels, _, _ = fgsm.generate(x_test, y_test)
np.savez('./FGSM_Test.npz', advs=advs, labels=labels)

advs, labels, _, _ = pgd.generate(x_test, y_test)
np.savez('./PGD_Test.npz', advs=advs, labels=labels)