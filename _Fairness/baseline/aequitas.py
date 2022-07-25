import sys
sys.path.append("../")

import os
import numpy as np
import random
from scipy.optimize import basinhopping
import tensorflow as tf
from tensorflow.python.platform import flags
import copy

from adf_data.census import census_data
from adf_data.credit import credit_data
from adf_data.bank import bank_data
from adf_utils.config import census, credit, bank
from adf_model.tutorial_models import dnn
from adf_utils.utils_tf import model_argmax

FLAGS = flags.FLAGS

class Local_Perturbation(object):
    """
    The  implementation of local perturbation
    """

    def __init__(self, sess, preds, x, conf, sensitive_param, param_probability, param_probability_change_size,
                 direction_probability, direction_probability_change_size, step_size):
        """
        Initial function of local perturbation
        :param sess: TF session
        :param preds: the model's symbolic output
        :param x: input placeholder
        :param conf: the configuration of dataset
        :param sensitive_param: the index of sensitive feature
        :param param_probability: the probabilities of features
        :param param_probability_change_size: the step size for changing probability
        :param direction_probability: the probabilities of perturbation direction
        :param direction_probability_change_size:
        :param step_size: the step size of perturbation
        """
        self.sess = sess
        self.preds = preds
        self.x = x
        self.conf = conf
        self.sensitive_param = sensitive_param
        self.param_probability = param_probability
        self.param_probability_change_size = param_probability_change_size
        self.direction_probability = direction_probability
        self.direction_probability_change_size = direction_probability_change_size
        self.step_size = step_size

    def __call__(self, x):
        """
        Local perturbation
        :param x: input instance for local perturbation
        :return: new potential individual discriminatory instance
        """
        # randomly choose the feature for perturbation
        param_choice = np.random.choice(range(self.conf.params) , p=self.param_probability)

        # randomly choose the direction for perturbation
        perturbation_options = [-1, 1]
        direction_choice = np.random.choice(perturbation_options, p=[self.direction_probability[param_choice],
                                                                     (1 - self.direction_probability[param_choice])])
        if (x[param_choice] == self.conf.input_bounds[param_choice][0]) or (x[param_choice] == self.conf.input_bounds[param_choice][1]):
            direction_choice = np.random.choice(perturbation_options)

        # perturbation
        x[param_choice] = x[param_choice] + (direction_choice * self.step_size)

        # clip the generating instance with each feature to make sure it is valid
        x[param_choice] = max(self.conf.input_bounds[param_choice][0], x[param_choice])
        x[param_choice] = min(self.conf.input_bounds[param_choice][1], x[param_choice])

        # check whether the test case is an individual discriminatory instance
        ei = check_for_error_condition(self.conf, self.sess, self.x, self.preds, x, self.sensitive_param)

        # update the probabilities of directions
        if (ei != int(x[self.sensitive_param - 1]) and direction_choice == -1) or (not (ei != int(x[self.sensitive_param - 1])) and direction_choice == 1):
            self.direction_probability[param_choice] = min(self.direction_probability[param_choice] +
                                                      (self.direction_probability_change_size * self.step_size), 1)
        elif (not (ei != int(x[self.sensitive_param - 1])) and direction_choice == -1) or (ei != int(x[self.sensitive_param - 1]) and direction_choice == 1):
            self.direction_probability[param_choice] = max(self.direction_probability[param_choice] -
                                                      (self.direction_probability_change_size * self.step_size), 0)

        # update the probabilities of features
        if ei != int(x[self.sensitive_param - 1]):
            self.param_probability[param_choice] = self.param_probability[param_choice] + self.param_probability_change_size
            self.normalise_probability()
        else:
            self.param_probability[param_choice] = max(self.param_probability[param_choice] - self.param_probability_change_size, 0)
            self.normalise_probability()

        return x

    def normalise_probability(self):
        """
        Normalize the probability
        :return: probability
        """
        probability_sum = 0.0
        for prob in self.param_probability:
            probability_sum = probability_sum + prob

        for i in range(self.conf.params):
            self.param_probability[i] = float(self.param_probability[i]) / float(probability_sum)


class Global_Discovery(object):
    """
    The  implementation of global perturbation
    """
    def __init__(self, conf):
        """
        Initial function of global perturbation
        :param conf: the configuration of dataset
        """
        self.conf = conf

    def __call__(self, x):
        """
        Global perturbation
        :param x: input instance for local perturbation
        :return: new potential individual discriminatory instance
        """
        # clip the generating instance with each feature to make sure it is valid
        for i in range(self.conf.params):
            x[i] = random.randint(self.conf.input_bounds[i][0], self.conf.input_bounds[i][1])
        return x

def check_for_error_condition(conf, sess, x, preds, t, sens):
    """
    Check whether the test case is an individual discriminatory instance
    :param conf: the configuration of dataset
    :param sess: TF session
    :param x: input placeholder
    :param preds: the model's symbolic output
    :param t: test case
    :param sens: the index of sensitive feature
    :return: the value of sensitive feature
    """
    t = np.array(t).astype("int")
    label = model_argmax(sess, x, preds, np.array([t]))

    # check for all the possible values of sensitive feature
    for val in range(conf.input_bounds[sens-1][0], conf.input_bounds[sens-1][1]+1):
        if val != int(t[sens-1]):
            tnew = copy.deepcopy(t)
            tnew[sens-1] = val
            label_new = model_argmax(sess, x, preds, np.array([tnew]))
            if label_new != label:
                return val
    return t[sens - 1]

def aequitas(dataset, sensitive_param, model_path, max_global, max_local, step_size):
    """
    The implementation of AEQUITAS_Fully_Connected
    :param dataset: the name of testing dataset
    :param sensitive_param: the name of testing dataset
    :param model_path: the path of testing model
    :param max_global: the maximum number of samples for global search
    :param max_local: the maximum number of samples for local search
    :param step_size: the step size of perturbation
    :return:
    """
    data = {"census": census_data, "credit": credit_data, "bank": bank_data}
    data_config = {"census": census, "credit": credit, "bank": bank}
    params = data_config[dataset].params

    # hyper-parameters for initial probabilities of directions
    init_prob = 0.5
    direction_probability = [init_prob] * params
    direction_probability_change_size = 0.001

    # hyper-parameters for features
    param_probability = [1.0 / params] * params
    param_probability_change_size = 0.001

    # prepare the testing data and model
    X, Y, input_shape, nb_classes = data[dataset]()
    model = dnn(input_shape, nb_classes)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    preds = model(x)
    tf.set_random_seed(1234)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    # store the result of fairness testing
    global_disc_inputs = set()
    global_disc_inputs_list = []
    local_disc_inputs = set()
    local_disc_inputs_list = []
    tot_inputs = set()

    # initial input
    if dataset == "census":
        initial_input = [7, 4, 26, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]
    elif dataset == "credit":
        initial_input = [2, 24, 2, 2, 37, 0, 1, 2, 1, 0, 4, 2, 2, 2, 1, 1, 2, 1, 0, 0]
    elif dataset == "bank":
        initial_input = [3, 11, 2, 0, 0, 5, 1, 0, 0, 5, 4, 40, 1, 1, 0, 0]
    minimizer = {"method": "L-BFGS-B"}

    def evaluate_local(inp):
        """
        Evaluate whether the test input after local perturbation is an individual discriminatory instance
        :param inp: test input
        :return: whether it is an individual discriminatory instance
        """
        result = check_for_error_condition(data_config[dataset], sess, x, preds, inp, sensitive_param)
        temp = copy.deepcopy(inp.astype('int').tolist())
        temp = temp[:sensitive_param - 1] + temp[sensitive_param:]
        tot_inputs.add(tuple(temp))
        if result != int(inp[sensitive_param - 1]) and (tuple(temp) not in global_disc_inputs) and (
            tuple(temp) not in local_disc_inputs):
            local_disc_inputs.add(tuple(temp))
            local_disc_inputs_list.append(temp)
        return not result

    global_discovery = Global_Discovery(data_config[dataset])
    local_perturbation = Local_Perturbation(sess, preds, x, data_config[dataset], sensitive_param, param_probability,
                                            param_probability_change_size, direction_probability,
                                            direction_probability_change_size, step_size)

    length = min(max_global, len(X))
    value_list = []
    for i in range(length):
        # global generation
        inp = global_discovery.__call__(initial_input)
        temp = copy.deepcopy(inp)
        temp = temp[:sensitive_param - 1] + temp[sensitive_param:]
        tot_inputs.add(tuple(temp))

        result = check_for_error_condition(data_config[dataset], sess, x, preds, inp, sensitive_param)

        # if get an individual discriminatory instance
        if result != inp[sensitive_param - 1] and (tuple(temp) not in global_disc_inputs) and (
            tuple(temp) not in local_disc_inputs):
            global_disc_inputs_list.append(temp)
            global_disc_inputs.add(tuple(temp))
            value_list.append([inp[sensitive_param - 1], result])

            # local generation
            basinhopping(evaluate_local, inp, stepsize=1.0, take_step=local_perturbation, minimizer_kwargs=minimizer,
                         niter=max_local)
            print(len(global_disc_inputs), len(local_disc_inputs),
                  "Percentage discriminatory inputs of local search- " + str(
                      float(len(local_disc_inputs)) / float(len(tot_inputs)) * 100))

    # create the folder for storing the fairness testing result
    if not os.path.exists('../results/'):
        os.makedirs('../results/')
    if not os.path.exists('../results/' + dataset + '/'):
        os.makedirs('../results/' + dataset + '/')
    if not os.path.exists('../results/'+ dataset + '/'+ str(sensitive_param) + '/'):
        os.makedirs('../results/' + dataset + '/'+ str(sensitive_param) + '/')

    # storing the fairness testing result
    np.save('../results/'+dataset+'/'+ str(sensitive_param) + '/global_samples_aequitas.npy', np.array(global_disc_inputs_list))
    np.save('../results/'+dataset+'/'+ str(sensitive_param) + '/disc_value_aequitas.npy', np.array(value_list))
    np.save('../results/' + dataset + '/' + str(sensitive_param) + '/local_samples_aequitas.npy', np.array(local_disc_inputs_list))

    # print the overview information of result
    print("Total Inputs are " + str(len(tot_inputs)))
    print("Total discriminatory inputs of global search- " + str(len(global_disc_inputs)))
    print("Total discriminatory inputs of local search- " + str(len(local_disc_inputs)))

def main(argv=None):
    aequitas(dataset = FLAGS.dataset,
             sensitive_param = FLAGS.sens_param,
             model_path = FLAGS.model_path,
             max_global = FLAGS.max_global,
             max_local = FLAGS.max_local,
             step_size = FLAGS.step_size)

if __name__ == '__main__':
    flags.DEFINE_string("dataset", "census", "the name of dataset")
    flags.DEFINE_integer('sens_param', 9, 'sensitive index, index start from 1, 9 for gender, 8 for race')
    flags.DEFINE_string('model_path', '../models/census/test.model', 'the path for testing model')
    flags.DEFINE_integer('max_global', 1000, 'number of maximum samples for global search')
    flags.DEFINE_integer('max_local', 1000, 'number of maximum samples for local search')
    flags.DEFINE_float('step_size', 1.0, 'step size for perturbation')

    tf.app.run()

