from tensorflow import keras 


def pair_loss(sess, x, preds, X_retrain_1, X_retrain_2):
    prd1 = model_prediction(sess, x, preds, X_retrain_1)
    prd2 = model_prediction(sess, x, preds, X_retrain_2)
    ce = tf.keras.losses.categorical_crossentropy(prd1, prd2)
    p_losses = sess.run(ce)
    return np.array(p_losses)


def fol_pair_loss(sess, x, preds, X_retrain_1, X_retrain_2):
    prd1 = model_prediction(sess, x, preds, X_retrain_1)
#     prd2 = model_prediction(sess, x, preds, X_retrain_2)
    ce = tf.keras.losses.categorical_crossentropy(prd1, preds)
    grad = tf.gradients(ce, x)
    grads = np.array(sess.run(grad, feed_dict = {x: X_retrain_2}))
    grads_norm = np.linalg.norm(grads.reshape(X_retrain_1.shape[0], -1), ord=2, axis=1)
    return np.array(grads_norm)

  
