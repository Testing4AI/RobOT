from tensorflow import keras
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.callbacks import ModelCheckpoint
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
        
    

def select(foscs, n, s='best', k=1000):
  
    ranks = np.argsort(foscs)
    # we choose test cases with small and large fols. 
    if s == 'best':
        h = n//2
        return np.concatenate((ranks[:h],ranks[-h:]))

    # we choose test cases with small and large fols.   
    elif s == 'kmst':
        index = []
        section_w = len(ranks) // k
        section_nums = n // section_w
        indexes = random.sample(list(range(k)), section_nums)
        for i in indexes:
            block = ranks[i*section_w: (i+1)*section_w]
            index.append(block)
        return np.concatenate(np.array(index))

    # This is for gini strategy. There is little different from DeepGini paper. See function ginis() in metrics.py 
    else:
        return ranks[:n]    
    

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
    
    
def load_mnist(path="./mnist.npz"):
    """
    preprocessing for MNIST dataset, values are normalized to [0,1].  
    y_train and y_test are one-hot vectors. 
    """
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


x_train, x_test, y_train, y_test = load_mnist(path="./mnist.npz")

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


sNums = [600*i for i in [1,2,3,4,6,8,10,12,16,20]]
strategies = ['best', 'kmst', 'gini']
acc_clean = [[] for i in range(len(strategies))]
acc_fp = [[] for i in range(len(strategies))]


for num in sNums:
    for i in range(len(strategies)):
        s = strategies[i]
        # model save path
        model_path = "./checkpoint/best_Lenet5_MIX_%d_%s.h5" % (num, s)
        model = keras.models.load_model("./Lenet5_mnist.h5")
        
        checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_accuracy', verbose=0, save_best_only=True)
        callbacks = [checkpoint]
        
        if s == 'gini':
            indexes = select(fp_train_ginis, num, s=s)
        else:
            indexes = select(fp_train_fols, num, s=s)

        selectAdvs = fp_train[indexes]
        selectAdvsLabels = fp_train_labels[indexes]
            
        x_train_mix = np.concatenate((x_train, selectAdvs),axis=0)
        y_train_mix = np.concatenate((y_train, selectAdvsLabels),axis=0)
        
        # model retraining 
        model.fit(x_train_mix, y_train_mix, epochs=10, batch_size=64, verbose=0, callbacks=callbacks,
                 validation_data=(fp_test, fp_test_labels))
        
        best_model = keras.models.load_model(model_path)
        _, aclean = best_model.evaluate(x_test, y_test, verbose=0)
        _, afp = best_model.evaluate(fp_test, fp_test_labels, verbose=0)
       
        acc_clean[i].append(aclean)
        acc_fp[i].append(afp)
        
        
        


        
        

        





