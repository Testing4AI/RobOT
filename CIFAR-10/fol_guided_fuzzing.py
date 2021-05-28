from tensorflow import keras
import random
import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.datasets import cifar10

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)



# cifar
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = keras.models.load_model("./saved_models/cifar10_resnet20_model.h5") 


seeds = random.sample(list(range(x_train.shape[0])), 1000)
images = x_train[seeds]
labels = y_train[seeds]


# some training samples is static, i.e., grad=<0>, hard to generate. 
seeds_filter = []
gen_img = tf.Variable(images)
with tf.GradientTape() as g:
    loss = keras.losses.categorical_crossentropy(labels, model(gen_img))
    grads = g.gradient(loss, gen_img)

fols = np.linalg.norm((grads.numpy()+1e-20).reshape(images.shape[0], -1), ord=2, axis=1)
seeds_filter = np.where(fols > 1e-3)[0]


start_t = time.time()
lr = 0.1
total_sets = []
for idx in seeds_filter:
    # delta_t = time.time() - start_t
    # if delta_t > 300:
    #     break
    img_list = []
    tmp_img = images[[idx]]
    orig_img = tmp_img.copy()
    orig_norm = np.linalg.norm(orig_img)
    img_list.append(tf.identity(tmp_img))
    logits = model(tmp_img)
    orig_index = np.argmax(logits[0])
    target = keras.utils.to_categorical([orig_index], 10)
    label_top5 = np.argsort(logits[0])[-5:]

    folMAX = 0 
    epoch = 0 
    while len(img_list) > 0:
        gen_img = img_list.pop(0)   
        for _ in range(2):
            gen_img = tf.Variable(gen_img)
            with tf.GradientTape(persistent=True) as g:
                loss = keras.losses.categorical_crossentropy(target, model(gen_img))
                grads = g.gradient(loss, gen_img)
                fol = tf.norm(grads+1e-20)
                g.watch(fol)
                logits = model(gen_img)
                obj = fol - logits[0][orig_index]
                dl_di = g.gradient(obj, gen_img)
            del g
            
            gen_img = gen_img + dl_di * lr * (random.random() + 0.5)
            gen_img = tf.clip_by_value(gen_img, clip_value_min=0, clip_value_max=1)
            
            with tf.GradientTape() as t:
                t.watch(gen_img)
                loss = keras.losses.categorical_crossentropy(target, model(gen_img))
                grad = t.gradient(loss, gen_img)
                fol = np.linalg.norm(grad.numpy()) # L2 adaption

            distance = np.linalg.norm(gen_img.numpy() - orig_img) / orig_norm
            if fol > folMAX and distance < 0.5:
                folMAX = fol
                img_list.append(tf.identity(gen_img))
            
            gen_index = np.argmax(model(gen_img)[0]) 
            if gen_index != orig_index:
                total_sets.append((fol, gen_img.numpy(), labels[idx]))


fols = np.array([item[0] for item in total_sets])
advs = np.array([item[1].reshape(32,32,3) for item in total_sets])
labels = np.array([item[2] for item in total_sets])

np.savez('./FOL_Fuzz.npz', advs=advs, labels=labels, fols=fols)


