import numpy as np
import sys
sys.path.append("../")

import tensorflow as tf
from tensorflow.python.platform import flags
from adf_data.census import census_data
from adf_data.bank import bank_data
from adf_data.credit import credit_data
from adf_utils.utils_tf import model_train, model_eval
from adf_model.tutorial_models import dnn

FLAGS = flags.FLAGS

def training(dataset, model_path, nb_epochs, batch_size,learning_rate):
    """
    Train the model
    :param dataset: the name of testing dataset
    :param model_path: the path to save trained model
    """
    data = {"census": census_data, "credit": credit_data, "bank": bank_data}

    # prepare the data and model
    X, Y, input_shape, nb_classes = data[dataset]()
    tf.set_random_seed(1234)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    model = dnn(input_shape, nb_classes)
    preds = model(x)

    # training parameters
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': model_path + dataset + "/",
        'filename': 'test.model'
    }

    # training procedure
    sess.run(tf.global_variables_initializer())
    rng = np.random.RandomState([2019, 7, 15])
    model_train(sess, x, y, preds, X, Y, args=train_params,
                rng=rng, save=True)

    # evaluate the accuracy of trained model
    eval_params = {'batch_size': 128}
    accuracy = model_eval(sess, x, y, preds, X, Y, args=eval_params)
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))

def main(argv=None):
    training(dataset = FLAGS.dataset,
             model_path = FLAGS.model_path,
             nb_epochs=FLAGS.nb_epochs,
             batch_size=FLAGS.batch_size,
             learning_rate=FLAGS.learning_rate)

if __name__ == '__main__':
    flags.DEFINE_string("dataset", "census", "the name of dataset")
    flags.DEFINE_string("model_path", "../models/", "the name of path for saving model")
    flags.DEFINE_integer('nb_epochs', 1000, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')

    tf.app.run()