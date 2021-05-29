from tensorflow import keras
import tensorflow as tf
import numpy as np


## Metrics for quality evaluation for massive test cases. 


def gini(model, x):
    """
    Different from the defination in DeepGini paper (deepgini = 1 - ginis), the smaller the ginis here, the larger the uncertainty. 
    
    shape of x: [batch_size, width, height, channel]
    """
    x = tf.Variable(x)
    preds = model(x).numpy()
    ginis = np.sum(np.square(preds), axis=1)
    return ginis
    
    
def fol_Linf(model, x, xi, ep, y):
    """
    x: perturbed inputs, shape of x: [batch_size, width, height, channel]   
    xi: initial inputs, shape of xi: [batch_size, width, height, channel]  
    ep: L_inf bound
    y: ground truth, one hot vectors, shape of y: [batch_size, N_classes] 
    """
    x, target = tf.Variable(x), tf.constant(y)
    fols = []
    with tf.GradientTape() as tape:
        loss = keras.losses.categorical_crossentropy(target, model(x))
        grads = tape.gradient(loss, x)
        grad_norm = np.linalg.norm(grads.numpy().reshape(x.shape[0], -1), ord=1, axis=1)
        grads_flat = grads.numpy().reshape(x.shape[0], -1)
        diff = (x.numpy() - xi).reshape(x.shape[0], -1)
        for i in range(x.shape[0]):
            i_fol = -np.dot(grads_flat[i], diff[i]) + ep * grad_norm[i]
            fols.append(i_fol)
    
    return np.array(fols)


def fol_L2(model, x, y):
    """
    x: perturbed inputs, shape of x: [batch_size, width, height, channel] 
    y: ground truth, one hot vectors, shape of y: [batch_size, N_classes] 
    """
    x, target = tf.Variable(x), tf.constant(y)
    with tf.GradientTape() as tape:
        loss = keras.losses.categorical_crossentropy(target, model(x))
        grads = tape.gradient(loss, x)
        grads_norm_L2 = np.linalg.norm(grads.numpy().reshape(x.shape[0], -1), ord=2, axis=1)

    return grads_norm_L2


def zol(model, x, y):
    """
    x: perturbed inputs, shape of x: [batch_size, width, height, channel] 
    y: ground truth, one hot vectors, shape of y: [batch_size, N_classes] 
    """
    x, target = tf.Variable(x), tf.constant(y)
    loss = keras.losses.categorical_crossentropy(target, model(x))
    loss.numpy().reshape(-1)

    return loss


def robustness(model, x, y):
    """
    x: perturbed inputs, shape of x: [batch_size, width, height, channel] 
    y: ground truth labels, shape of y: [batch_size] 
    """
    return np.sum(np.argmax(model(x), axis=1) == y) / y.shape[0]

