import numpy as np
import tensorflow as tf
import sys, os
sys.path.append("../")
import copy

from tensorflow.python.platform import flags
from scipy.optimize import basinhopping

from adf_data.census import census_data
from adf_data.credit import credit_data
from adf_data.bank import bank_data
from adf_model.tutorial_models import dnn
from adf_utils.utils_tf import model_prediction, model_argmax
from adf_utils.config import census, credit, bank
from adf_tutorial.utils import cluster, gradient_graph

FLAGS = flags.FLAGS

# step size of perturbation
perturbation_size = 1

def check_for_error_condition(conf, sess, x, preds, t, sens):
    """
    Check whether the test case is an individual discriminatory instance
    :param conf: the configuration of dataset
    :param sess: TF session
    :param x: input placeholder
    :param preds: the model's symbolic output
    :param t: test case
    :param sens: the index of sensitive feature
    :return: whether it is an individual discriminatory instance
    """
    t = t.astype('int')
    label = model_argmax(sess, x, preds, np.array([t]))

    # check for all the possible values of sensitive feature
    for val in range(conf.input_bounds[sens-1][0], conf.input_bounds[sens-1][1]+1):
        if val != t[sens-1]:
            tnew = copy.deepcopy(t)
            tnew[sens-1] = val
            label_new = model_argmax(sess, x, preds, np.array([tnew]))
            if label_new != label:
                return True
    return False

def seed_test_input(clusters, limit):
    """
    Select the seed inputs for fairness testing
    :param clusters: the results of K-means clustering
    :param limit: the size of seed inputs wanted
    :return: a sequence of seed inputs
    """
    i = 0
    rows = []
    max_size = max([len(c[0]) for c in clusters])
    while i < max_size:
        if len(rows) == limit:
            break
        for c in clusters:
            if i >= len(c[0]):
                continue
            row = c[0][i]
            rows.append(row)
            if len(rows) == limit:
                break
        i += 1
    return np.array(rows)

def clip(input, conf):
    """
    Clip the generating instance with each feature to make sure it is valid
    :param input: generating instance
    :param conf: the configuration of dataset
    :return: a valid generating instance
    """
    for i in range(len(input)):
        input[i] = max(input[i], conf.input_bounds[i][0])
        input[i] = min(input[i], conf.input_bounds[i][1])
    return input

class Local_Perturbation(object):
    """
    The  implementation of local perturbation
    """

    def __init__(self, sess, grad, x, n_value, sens, input_shape, conf):
        """
        Initial function of local perturbation
        :param sess: TF session
        :param grad: the gradient graph
        :param x: input placeholder
        :param n_value: the discriminatory value of sensitive feature
        :param sens_param: the index of sensitive feature
        :param input_shape: the shape of dataset
        :param conf: the configuration of dataset
        """
        self.sess = sess
        self.grad = grad
        self.x = x
        self.n_value = n_value
        self.input_shape = input_shape
        self.sens = sens
        self.conf = conf

    def __call__(self, x):
        """
        Local perturbation
        :param x: input instance for local perturbation
        :return: new potential individual discriminatory instance
        """

        # perturbation
        s = np.random.choice([1.0, -1.0]) * perturbation_size

        n_x = x.copy()
        n_x[self.sens - 1] = self.n_value

        # compute the gradients of an individual discriminatory instance pairs
        ind_grad = self.sess.run(self.grad, feed_dict={self.x:np.array([x])})
        n_ind_grad = self.sess.run(self.grad, feed_dict={self.x:np.array([n_x])})

        if np.zeros(self.input_shape).tolist() == ind_grad[0].tolist() and np.zeros(self.input_shape).tolist() == \
                n_ind_grad[0].tolist():
            probs = 1.0 / (self.input_shape-1) * np.ones(self.input_shape)
            probs[self.sens - 1] = 0
        else:
            # nomalize the reciprocal of gradients (prefer the low impactful feature)
            grad_sum = 1.0 / (abs(ind_grad[0]) + abs(n_ind_grad[0]))
            grad_sum[self.sens - 1] = 0
            probs = grad_sum / np.sum(grad_sum)
        probs = probs/probs.sum()

        # randomly choose the feature for local perturbation
        index = np.random.choice(range(self.input_shape) , p=probs)
        local_cal_grad = np.zeros(self.input_shape)
        local_cal_grad[index] = 1.0

        x = clip(x + s * local_cal_grad, self.conf).astype("int")

        return x

def dnn_fair_testing(dataset, sensitive_param, model_path, cluster_num, max_global, max_local, max_iter):
    """
    The implementation of ADF
    :param dataset: the name of testing dataset
    :param sensitive_param: the index of sensitive feature
    :param model_path: the path of testing model
    :param cluster_num: the number of clusters to form as well as the number of
            centroids to generate
    :param max_global: the maximum number of samples for global search
    :param max_local: the maximum number of samples for local search
    :param max_iter: the maximum iteration of global perturbation
    """
    data = {"census":census_data, "credit":credit_data, "bank":bank_data}
    data_config = {"census":census, "credit":credit, "bank":bank}

    # prepare the testing data and model
    X, Y, input_shape, nb_classes = data[dataset]()
    tf.set_random_seed(1234)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    model = dnn(input_shape, nb_classes)
    preds = model(x)
    saver = tf.train.Saver()
    model_path = model_path + dataset + "/test.model"
    saver.restore(sess, model_path)

    # construct the gradient graph
    grad_0 = gradient_graph(x, preds)

    # build the clustering model
    clf = cluster(dataset, cluster_num)
    clusters = [np.where(clf.labels_==i) for i in range(cluster_num)]

    # store the result of fairness testing
    tot_inputs = set()
    global_disc_inputs = set()
    global_disc_inputs_list = []
    local_disc_inputs = set()
    local_disc_inputs_list = []
    value_list = []
    suc_idx = []

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
        if result and (tuple(temp) not in global_disc_inputs) and (tuple(temp) not in local_disc_inputs):
            local_disc_inputs.add(tuple(temp))
            local_disc_inputs_list.append(temp)

        return not result

    # select the seed input for fairness testing
    inputs = seed_test_input(clusters, min(max_global, len(X)))

    for num in range(len(inputs)):
        index = inputs[num]
        sample = X[index:index+1]

        # start global perturbation
        for iter in range(max_iter+1):
            probs = model_prediction(sess, x, preds, sample)[0]
            label = np.argmax(probs)
            prob = probs[label]
            max_diff = 0
            n_value = -1

            # search the instance with maximum probability difference for global perturbation
            for i in range(census.input_bounds[sensitive_param-1][0], census.input_bounds[sensitive_param-1][1] + 1):
                if i != sample[0][sensitive_param-1]:
                    n_sample = sample.copy()
                    n_sample[0][sensitive_param-1] = i
                    n_probs = model_prediction(sess, x, preds, n_sample)[0]
                    n_label = np.argmax(n_probs)
                    n_prob = n_probs[n_label]
                    if label != n_label:
                        n_value = i
                        break
                    else:
                        prob_diff = abs(prob - n_prob)
                        if prob_diff > max_diff:
                            max_diff = prob_diff
                            n_value = i

            temp = copy.deepcopy(sample[0].astype('int').tolist())
            temp = temp[:sensitive_param - 1] + temp[sensitive_param:]

            # if get an individual discriminatory instance
            if label != n_label and (tuple(temp) not in global_disc_inputs) and (tuple(temp) not in local_disc_inputs):
                global_disc_inputs_list.append(temp)
                global_disc_inputs.add(tuple(temp))
                value_list.append([sample[0, sensitive_param - 1], n_value])
                suc_idx.append(index)
                print(len(suc_idx), num)

                # start local perturbation
                minimizer = {"method": "L-BFGS-B"}
                local_perturbation = Local_Perturbation(sess, grad_0, x, n_value, sensitive_param, input_shape[1],
                                                        data_config[dataset])
                basinhopping(evaluate_local, sample, stepsize=1.0, take_step=local_perturbation,
                             minimizer_kwargs=minimizer,
                             niter=max_local)

                print(len(local_disc_inputs_list),
                      "Percentage discriminatory inputs of local search- " + str(
                          float(len(local_disc_inputs)) / float(len(tot_inputs)) * 100))
                break

            n_sample[0][sensitive_param - 1] = n_value

            if iter == max_iter:
                break

            # global perturbation
            s_grad = sess.run(tf.sign(grad_0), feed_dict={x: sample})
            n_grad = sess.run(tf.sign(grad_0), feed_dict={x: n_sample})

            # find the feature with same impact
            if np.zeros(data_config[dataset].params).tolist() == s_grad[0].tolist():
                g_diff = n_grad[0]
            elif np.zeros(data_config[dataset].params).tolist() == n_grad[0].tolist():
                g_diff = s_grad[0]
            else:
                g_diff = np.array(s_grad[0] == n_grad[0], dtype=float)
            g_diff[sensitive_param - 1] = 0
            if np.zeros(input_shape[1]).tolist() == g_diff.tolist():
                index = np.random.randint(len(g_diff) - 1)
                if index > sensitive_param - 2:
                    index = index + 1
                g_diff[index] = 1.0

            cal_grad = s_grad * g_diff
            sample[0] = clip(sample[0] + perturbation_size * cal_grad[0], data_config[dataset]).astype("int")

    # create the folder for storing the fairness testing result
    if not os.path.exists('../results/'):
        os.makedirs('../results/')
    if not os.path.exists('../results/' + dataset + '/'):
        os.makedirs('../results/' + dataset + '/')
    if not os.path.exists('../results/'+ dataset + '/'+ str(sensitive_param) + '/'):
        os.makedirs('../results/' + dataset + '/'+ str(sensitive_param) + '/')

    # storing the fairness testing result
    np.save('../results/'+dataset+'/'+ str(sensitive_param) + '/suc_idx.npy', np.array(suc_idx))
    np.save('../results/'+dataset+'/'+ str(sensitive_param) + '/global_samples.npy', np.array(global_disc_inputs_list))
    np.save('../results/'+dataset+'/'+ str(sensitive_param) + '/local_samples.npy', np.array(local_disc_inputs_list))
    np.save('../results/'+dataset+'/'+ str(sensitive_param) + '/disc_value.npy', np.array(value_list))

    # print the overview information of result
    print("Total Inputs are " + str(len(tot_inputs)))
    print("Total discriminatory inputs of global search- " + str(len(global_disc_inputs)))
    print("Total discriminatory inputs of local search- " + str(len(local_disc_inputs)))

def main(argv=None):
    dnn_fair_testing(dataset = FLAGS.dataset,
                     sensitive_param = FLAGS.sens_param,
                     model_path = FLAGS.model_path,
                     cluster_num=FLAGS.cluster_num,
                     max_global=FLAGS.max_global,
                     max_local=FLAGS.max_local,
                     max_iter = FLAGS.max_iter)

if __name__ == '__main__':
    flags.DEFINE_string("dataset", "census", "the name of dataset")
    flags.DEFINE_integer('sens_param', 9, 'sensitive index, index start from 1, 9 for gender, 8 for race')
    flags.DEFINE_string('model_path', '../models/', 'the path for testing model')
    flags.DEFINE_integer('cluster_num', 4, 'the number of clusters to form as well as the number of centroids to generate')
    flags.DEFINE_integer('max_global', 1000, 'maximum number of samples for global search')
    flags.DEFINE_integer('max_local', 1000, 'maximum number of samples for local search')
    flags.DEFINE_integer('max_iter', 10, 'maximum iteration of global perturbation')

    tf.app.run()
