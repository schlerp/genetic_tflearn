import random
import itertools
import tensorflow
import tflearn
from tflearn.data_utils import *


# load data
X, Y, Xval, Yval = tflearn.datasets.mnist.load_data(one_hot=True)
# shorten training sets cus its just POC at this stage
#X = X[0:20000]
#Y = Y[0:20000]
#Xval = Xval[0:5000]
#Yval = Yval[0:5000]
# reshape to tensors
X = X.reshape([-1, 28, 28, 1])
Xval = Xval.reshape([-1, 28, 28, 1])

# set input/output shapes
in_shape = [None, 28, 28, 1]
out_nodes = 10


def get_initializer(init_name):
    return tflearn.initializations.get(init_name)

def make_model(conv_layer, conv_filter, conv_kern, conv_activation, conv_regularizer, conv_w_initializer, conv_b_initializer, conv_w_seed, conv_b_seed,
               max_pool, 
               fc_layer, fc_width, fc_activation, fc_regularizer, fc_w_initializer, fc_b_initializer, fc_w_seed, fc_b_seed, learn_rate, optimizer):
    # input layer
    network = tflearn.input_data(shape=in_shape, name='input')

    # conv layers
    for _ in range(conv_layer):
        conv_w_init = get_initializer(conv_w_initializer)
        conv_b_init = get_initializer(conv_b_initializer)
        network = tflearn.conv_2d(network, conv_filter, conv_kern, 
                                  activation=conv_activation, 
                                  regularizer=conv_regularizer,
                                  weights_init=conv_w_init(seed=conv_w_seed),
                                  bias_init=conv_b_init(seed=conv_b_seed))

    # max pool layers
    if max_pool > 0:
        network = tflearn.max_pool_2d(network, max_pool)

    # linear layers
    for _ in range(fc_layer):
        fc_w_init = get_initializer(fc_w_initializer)
        fc_b_init = get_initializer(fc_b_initializer)
        network = tflearn.fully_connected(network, fc_width, 
                                          activation=fc_activation,
                                          regularizer=fc_regularizer,
                                          weights_init=fc_w_init(seed=fc_w_seed),
                                          bias_init=fc_b_init(seed=fc_b_seed))

    # output layer
    network = tflearn.fully_connected(network, out_nodes, activation='softmax')
    network = tflearn.regression(network, optimizer=optimizer, learning_rate=learn_rate,
                                 loss='categorical_crossentropy', name='target')
    return network


if __name__ == '__main__': 
    test1 = [('conv_layers', 2),
             ('conv_filters', 24),
             ('conv_kern', 4),
             ('conv_activations', 'linear'),
             ('conv_regularizer', 'L1'),
             ('conv_w_initializer', 'xavier'),
             ('conv_b_initializer', 'variance_scaling'),
             ('conv_w_seed', 26099),
             ('conv_b_seed', 5796),
             ('max_pool', 1),
             ('fc_layers', 1),
             ('fc_widths', 96),
             ('fc_activations', 'elu'),
             ('fc_regularizer', 'L2'),
             ('fc_w_initializer', 'truncated_normal'),
             ('fc_b_initializer', 'variance_scaling'),
             ('fc_w_seed', 11188),
             ('fc_b_seed', 43068),
             ('learn_rates', 0.001),
             ('optimizers', 'adam')]
    
    test2 = [('conv_layers', 1),
             ('conv_filters', 32),
             ('conv_kern', 2),
             ('conv_activations', 'tanh'),
             ('conv_regularizer', False),
             ('conv_w_initializer', 'uniform_scaling'),
             ('conv_b_initializer', 'variance_scaling'),
             ('conv_w_seed', 41625),
             ('conv_b_seed', 23625),
             ('max_pool', 0),
             ('fc_layers', 1),
             ('fc_widths', 176),
             ('fc_activations', 'sigmoid'),
             ('fc_regularizer', False),
             ('fc_w_initializer', 'variance_scaling'),
             ('fc_b_initializer', 'xavier'),
             ('fc_w_seed', 51215),
             ('fc_b_seed', 38759),
             ('learn_rates', 0.1),
             ('optimizers', 'momentum')]
    
    
    net = make_model(*[x[1] for x in test2])
    
    model = tflearn.DNN(net, tensorboard_verbose=0)
    
    model.fit(X, Y, n_epoch=10, show_metric=False,
              validation_set=(Xval, Yval),
              run_id='ga_convnet_mnist')
    score = model.evaluate(Xval, Yval)
    print(score)
    
    test1_val = [0.97860000000000003]