from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Load the generated adversarial inputs for Robustness evaluation. 
with np.load("./FGSM_Test.npz") as f:
    fgsm_test, fgsm_test_labels = f['advs'], f['labels']

with np.load("./PGD_Test.npz") as f:
    pgd_test, pgd_test_labels = f['advs'], f['labels']

fp_test = np.concatenate((fgsm_test, pgd_test))
fp_test_labels = np.concatenate((fgsm_test_labels, pgd_test_labels))


sNums = [500*i for i in [2,4,8,12,20]]
strategies = ['best', 'kmst', 'gini']
acc_pure = [[] for i in range(len(strategies))]
acc_fp = [[] for i in range(len(strategies))]


for num in sNums:
    for i in range(len(strategies)):
        s = strategies[i]
        model_path = "./checkpoint/best_Resnet_MIX_%d_%s.h5" % (num, s)
        best_model = keras.models.load_model(model_path)
        lfp, afp = best_model.evaluate(fp_test, fp_test_labels, verbose=0)
        acc_fp[i].append(afp)



colormap = ['r','limegreen', 'dodgerblue']
plt.figure(figsize=(8,6))
x = [i/max(sNums) for i in sNums]
for i in range(len(strategies)):
    plt.plot(x, acc_fp[i],'o-', label=strategies[i], color=colormap[i], linewidth=3, markersize=8)

plt.title("CIFAR-ATTACK", fontsize=20)
plt.xlabel("# Percentage of test cases", fontsize=20)
plt.ylabel("Robustness", fontsize=20)
plt.xticks(x, [1,2,4,6,10],fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)

fig = plt.gcf()
fig.savefig('./cifar_attack_robustness.pdf')

