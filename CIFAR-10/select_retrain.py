from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
import random

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

# Suppress the GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
    

def lr_schedule_retrain(epoch):
    lr = 1e-4
    if epoch > 25:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr



def select(values, n, s='best', k=4):
    """
    n: the number of selected test cases. 
    s: strategy, ['best', 'random', 'kmst', 'gini']
    k: for KM-ST, the number of ranges. 
    """
    ranks = np.argsort(values) 
    
    if s == 'best':
        h = n//2
        return np.concatenate((ranks[:h],ranks[-h:]))
        
    elif s == 'r':
        return np.array(random.sample(list(ranks),n)) 
    
    elif s == 'kmst':
        fol_max = values.max()
        th = fol_max / k
        section_nums = n // k
        indexes = []
        for i in range(k):
            section_indexes = np.intersect1d(np.where(values<th*(i+1)), np.where(values>=th*i))
            if section_nums < len(section_indexes):
                index = random.sample(list(section_indexes), section_nums)
                indexes.append(index)
            else: 
                indexes.append(section_indexes)
                index = random.sample(list(ranks), section_nums-len(section_indexes))
                indexes.append(index)
        return np.concatenate(np.array(indexes))

    # This is for gini strategy. There is little difference from DeepGini paper. See function ginis() in metrics.py 
    else: 
        return ranks[:n]   
    
    

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# convert class vectors to binary class matrics
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
x_train = x_train/255
x_test = x_test/255


# Load the generated adversarial inputs for training. FGSM and PGD. 
with np.load("./FGSM_TrainFull.npz") as f:
    fgsm_train, fgsm_train_labels, fgsm_train_fols, fgsm_train_ginis = f['advs'], f['labels'], f['fols'], f['ginis']
    
with np.load("./PGD_TrainFull.npz") as f:
    pgd_train, pgd_train_labels, pgd_train_fols, pgd_train_ginis= f['advs'], f['labels'], f['fols'], f['ginis']
    
# Load the generated adversarial inputs for testing. FGSM and PGD. 
with np.load("./FGSM_Test.npz") as f:
    fgsm_test, fgsm_test_labels = f['advs'], f['labels']

with np.load("./PGD_Test.npz") as f:
    pgd_test, pgd_test_labels = f['advs'], f['labels']


# Mix the adversarial inputs 
fp_train = np.concatenate((fgsm_train, pgd_train))
fp_train_labels = np.concatenate((fgsm_train_labels, pgd_train_labels))
fp_train_fols = np.concatenate((fgsm_train_fols, pgd_train_fols))
fp_train_ginis = np.concatenate((fgsm_train_ginis, pgd_train_ginis))

fp_test = np.concatenate((fgsm_test, pgd_test))
fp_test_labels = np.concatenate((fgsm_test_labels, pgd_test_labels))


sNums = [500*i for i in [2,4,8,12,20]]
strategies = ['best', 'kmst', 'gini']

for num in sNums:
    print(num)
    for i in range(len(strategies)):
        s = strategies[i]
        model_path = "./checkpoint/best_Resnet_MIX_%d_%s.h5" % (num, s)
        checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_accuracy', verbose=1, save_best_only=True)
        lr_scheduler = LearningRateScheduler(lr_schedule_retrain)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        callbacks = [checkpoint, lr_reducer, lr_scheduler]
        
        if s == 'gini':
            indexes = select(fp_train_ginis, num, s=s)
        else:
            indexes = select(fp_train_fols, num, s=s)

        selectAdvs = fp_train[indexes]
        selectAdvsLabels = fp_train_labels[indexes]
            
        x_train_mix = np.concatenate((x_train, selectAdvs),axis=0)
        y_train_mix = np.concatenate((y_train, selectAdvsLabels),axis=0)
        

        # load old model 
        model = keras.models.load_model("./saved_models/cifar10_resnet20_model.h5")  
#         model.fit(x_train_mix, y_train_mix, epochs=40, batch_size=64, verbose=1, callbacks=callbacks,
#                  validation_data=(fp_test, fp_test_labels))

        datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        datagen.fit(x_train_mix)
        batch_size = 64
        history = model.fit_generator(datagen.flow(x_train_mix, y_train_mix, batch_size=batch_size),
                            validation_data=(fp_test, fp_test_labels),
                            epochs=40, verbose=1,
                            callbacks=callbacks,
                            steps_per_epoch= x_train_mix.shape[0] // batch_size)


        
        
        


        
        

        





