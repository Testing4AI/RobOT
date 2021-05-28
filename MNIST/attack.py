from tensorflow import keras
import tensorflow as tf
import numpy as np



class FGSM:
    """
    We use FGSM to generate a batch of adversarial examples. 
    """
    def __init__(self, model, ep=0.3, isRand=True):
        """
        isRand is set True to improve the attack success rate. 
        """
        self.isRand = isRand
        self.model = model
        self.ep = ep
        
    def generate(self, x, y, randRate=1):
        """
        x: clean inputs, shape of x: [batch_size, width, height, channel] 
        y: ground truth, one hot vectors, shape of y: [batch_size, N_classes] 
        """
        fols = []
        target = tf.constant(y)
        
        xi = x.copy()
        if self.isRand:
            x = x + np.random.uniform(-self.ep * randRate, self.ep * randRate, x.shape)
            x = np.clip(x, 0, 1)
        
        x = tf.Variable(x)
        with tf.GradientTape() as tape:
            loss = keras.losses.categorical_crossentropy(target, self.model(x))
            grads = tape.gradient(loss, x)
        delta = tf.sign(grads)
        x_adv = x + self.ep * delta
        
        x_adv = tf.clip_by_value(x_adv, clip_value_min=xi-self.ep, clip_value_max=xi+self.ep)
        x_adv = tf.clip_by_value(x_adv, clip_value_min=0, clip_value_max=1)
        
        idxs = np.where(np.argmax(self.model(x_adv), axis=1) != np.argmax(y, axis=1))[0]
        print("SUCCESS:", len(idxs))
        
        x_adv, xi, target = x_adv.numpy()[idxs], xi[idxs], target.numpy()[idxs]
        x_adv, target = tf.Variable(x_adv), tf.constant(target)
        
        preds = self.model(x_adv).numpy()
        ginis = np.sum(np.square(preds), axis=1)
        
        with tf.GradientTape() as tape:
            loss = keras.losses.categorical_crossentropy(target, self.model(x_adv))
            grads = tape.gradient(loss, x_adv)
            grad_norm = np.linalg.norm(grads.numpy().reshape(x_adv.shape[0], -1), ord=1, axis=1)
            grads_flat = grads.numpy().reshape(x_adv.shape[0], -1)
            diff = (x_adv.numpy() - xi).reshape(x_adv.shape[0], -1)
            for i in range(x_adv.shape[0]):
                i_fol = -np.dot(grads_flat[i], diff[i]) + self.ep * grad_norm[i]
                fols.append(i_fol)
  
        return x_adv.numpy(), target.numpy(), np.array(fols), ginis



class PGD:
    """
    We use PGD to generate a batch of adversarial examples. PGD could be seen as iterative version of FGSM.
    """
    def __init__(self, model, ep=0.3, step=None, epochs=10, isRand=True):
        """
        isRand is set True to improve the attack success rate. 
        """
        self.isRand = isRand
        self.model = model
        self.ep = ep
        if step == None:
            self.step = ep/6
        self.epochs = epochs
        
    def generate(self, x, y, randRate=1):
        """
        x: clean inputs, shape of x: [batch_size, width, height, channel] 
        y: ground truth, one hot vectors, shape of y: [batch_size, N_classes] 
        """
        fols = []
        target = tf.constant(y)
    
        xi = x.copy()
        if self.isRand:
            x = x + np.random.uniform(-self.ep * randRate, self.ep * randRate, x.shape)
            x = np.clip(x, 0, 1)
        
        x_adv = tf.Variable(x)
        for i in range(self.epochs): 
            with tf.GradientTape() as tape:
                loss = keras.losses.categorical_crossentropy(target, self.model(x_adv))
                grads = tape.gradient(loss, x_adv)
            delta = tf.sign(grads)
            x_adv.assign_add(self.step * delta)
            x_adv = tf.clip_by_value(x_adv, clip_value_min=xi-self.ep, clip_value_max=xi+self.ep)
            x_adv = tf.clip_by_value(x_adv, clip_value_min=0, clip_value_max=1)
            x_adv = tf.Variable(x_adv)
        
        idxs = np.where(np.argmax(self.model(x_adv), axis=1) != np.argmax(y, axis=1))[0]
        print("SUCCESS:", len(idxs))
        
        x_adv, xi, target = x_adv.numpy()[idxs], xi[idxs], target.numpy()[idxs]
        x_adv, target = tf.Variable(x_adv), tf.constant(target)
        
        preds = self.model(x_adv).numpy()
        ginis = np.sum(np.square(preds), axis=1)
        
        with tf.GradientTape() as tape:
            loss = keras.losses.categorical_crossentropy(target, self.model(x_adv))
            grads = tape.gradient(loss, x_adv)
            grad_norm = np.linalg.norm(grads.numpy().reshape(x_adv.shape[0], -1), ord=1, axis=1)
            grads_flat = grads.numpy().reshape(x_adv.shape[0], -1)
            diff = (x_adv.numpy() - xi).reshape(x_adv.shape[0], -1)
            for i in range(x_adv.shape[0]):
                i_fol = -np.dot(grads_flat[i], diff[i]) + self.ep * grad_norm[i]
                fols.append(i_fol)
  
        return x_adv.numpy(), target.numpy(), np.array(fols), ginis
    

  
    