import random
import itertools
import tensorflow
import tflearn
from tflearn.data_utils import *


# load data
X, Y, Xval, Yval = tflearn.datasets.mnist.load_data(one_hot=True)
# shorten training sets cus its just POC at this stage
X_test = X[0:2000]
Y_test = Y[0:2000]
Xval_test = Xval[0:500]
Yval_test = Yval[0:500]
# reshape to tensors
X = X.reshape([-1, 28, 28, 1])
X_test = X_test.reshape([-1, 28, 28, 1])
Xval = Xval.reshape([-1, 28, 28, 1])
Xval_test = Xval_test.reshape([-1, 28, 28, 1])

# set input/output shapes
in_shape = [None, 28, 28, 1]
out_nodes = 10

# other settings
pop_size = 15
num_epoch = 5
mut_chance = 50 # percentage chance to mutate

# global vars
best_genetics = (0, None)

seed_space = [x for x in range(0, 65535)]
activation_space = ['linear', 'relu', 'elu', 'tanh', 'sigmoid']
initializer_space = ['zeros', 'uniform', 'uniform_scaling', 'normal', 
                     'truncated_normal', 'xavier', 'variance_scaling']
regularizer_space = [None, 'L1', 'L2']


hps = {'conv_layers': [x for x in range(1, 5)],
       'conv_filters': [x for x in range(4, 128, 4)],
       'conv_kern': [x for x in range(1, 5)], 
       'conv_activations': activation_space, 
       'conv_regularizer': regularizer_space, 
       'conv_w_initializer': initializer_space, 
       'conv_b_initializer': initializer_space, 
       'conv_w_seed': seed_space, 
       'conv_b_seed': seed_space, 
       
       'conv_blocks': [x for x in range(1, 3)],
       
       'max_pool': [None, 2, 3, 4, 5], 
       
       'fc_layers': [x for x in range(1, 10)],
       'fc_widths': [x for x in range(32, 512, 16)],
       'fc_activations': activation_space,
       'fc_regularizer': regularizer_space,
       'fc_w_initializer': initializer_space,
       'fc_b_initializer': initializer_space,
       'fc_w_seed': seed_space,
       'fc_b_seed': seed_space,
       
       'learn_rates': [0.0001, 0.001, 0.01, 0.1, 1],
       'optimizers': ['sgd', 'rmsprop', 'adam', 'momentum', 'adagrad'],
       }

hps_map = ('conv_layers', 'conv_filters', 'conv_kern', 'conv_activations', 
           'conv_regularizer', 'conv_w_initializer', 'conv_b_initializer', 'conv_w_seed', 'conv_b_seed', 
           'conv_blocks', 'max_pool', 
           'fc_layers', 'fc_widths', 'fc_activations', 
           'fc_regularizer', 'fc_w_initializer', 'fc_b_initializer', 'fc_w_seed', 'fc_b_seed', 
           'learn_rates', 'optimizers')


def breed(parent1, parent2):
    crossover = random.randint(1, len(parent2)-2)
    child = mutate(parent1[0:crossover] + parent2[crossover:])
    return child

def mutate(child):
    if random.randint(0, 100) <= mut_chance:
        for _ in range(random.randint(1, len(hps_map)//2)):
            gene = random.randint(0, len(hps_map)-1)
            gene_name = hps_map[gene]
            child[gene] = random.choice(hps[gene_name])
    return child

def get_initializer(init_name):
    return tflearn.initializations.get(init_name)

def make_model(conv_layer, conv_filter, conv_kern, conv_activation, conv_regularizer, conv_w_initializer, conv_b_initializer, conv_w_seed, conv_b_seed,
               conv_blocks, max_pool, 
               fc_layer, fc_width, fc_activation, fc_regularizer, fc_w_initializer, fc_b_initializer, fc_w_seed, fc_b_seed, learn_rate, optimizer):
    # input layer
    network = tflearn.input_data(shape=in_shape, name='input')
    
    # conv layers
    for i in range(conv_blocks):
        for j in range(conv_layer):
            conv_w_init = get_initializer(conv_w_initializer)
            conv_b_init = get_initializer(conv_b_initializer)
            network = tflearn.conv_2d(network, conv_filter, conv_kern, 
                                      activation=conv_activation, 
                                      regularizer=conv_regularizer,
                                      weights_init=conv_w_init(seed=conv_w_seed),
                                      bias_init=conv_b_init(seed=conv_b_seed))
        
        # max pool layers
        if max_pool:
            network = tflearn.max_pool_2d(network, max_pool)
    
    # linear layers
    for k in range(fc_layer):
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

def init_pop():
    entities = []
    for _ in range(pop_size):
        genes = []
        for gene_name in hps_map:
            genes.append(random.choice(hps[gene_name]))
        
        entities.append(genes)
        #entities.append(make_model(*genes))
    return entities

def handle_breed(top3):
    new_pop = [x[1] for x in top3]
    combinations = itertools.combinations(top3, 2)
    for parents in combinations:
        for _ in range(2):
            new_pop.append(breed(parents[0][1], parents[1][1]))
    return new_pop

def print_best():
    print("best network so far scored {}".format(best_genetics[0]))
    print("params:")
    net_def = zip(hps_map, best_genetics[1])
    for item in net_def:
        print(item)    

def main():
    try:
        pop = init_pop()
        generation = 1
        while True:
            print('Training Generation {}'.format(generation))
            results = []
            for key, genes in enumerate(pop):
                with tensorflow.Graph().as_default():
                    
                    print('network {}'.format(key))
                    net_def = zip(hps_map, genes)
                    for item in net_def:
                        print(item)
                        
                    net = make_model(*genes)
                    model = tflearn.DNN(net, tensorboard_verbose=0)
                    model.fit(X_test, Y_test, n_epoch=num_epoch, show_metric=False, 
                              run_id='ga_convnet_mnist')
                    
                    score = model.evaluate(Xval_test, Yval_test)[0]
                    print(score)
                    results.append((score, genes))
                    
            results.sort(key=lambda res: res[0], reverse=True)
            
            print(results[0][0])
            global best_genetics
            if results[0][0] > best_genetics[0]:
                best_genetics = results[0]
                
            top3 = results[0:3]
            print(top3)
            pop = handle_breed(top3)
            # allow a shitter one through to help stop getting stuck at local maximum
            pop.append(results[5][1])
            print_best()
            print('end of generation {}'.format(generation))
            generation += 1

    except KeyboardInterrupt:
        print_best()
        print("Testing best network...")
        genes = best_genetics[1]
        net = make_model(*genes)
        model = tflearn.DNN(net, tensorboard_verbose=0)
    
        model.fit(X, Y, n_epoch=10, show_metric=False,
                  validation_set=(Xval, Yval),
                  run_id='ga_convnet_mnist')
        score = model.evaluate(Xval, Yval)
        print(score)


if __name__ == '__main__':
    main()
