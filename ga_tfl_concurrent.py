import random
import itertools
import tensorflow
import tflearn
import multiprocessing as mp
from tflearn.data_utils import *

# load data
X, Y, Xval, Yval = tflearn.datasets.mnist.load_data(one_hot=True)
# shorten training sets cus its just POC at this stage
X = X[0:5000]
Y = Y[0:5000]
Xval = Xval[0:1000]
Yval = Yval[0:1000]
# reshape to tensors
X = X.reshape([-1, 28, 28, 1])
Xval = Xval.reshape([-1, 28, 28, 1])

# set input/output shapes
in_shape = [None, 28, 28, 1]
out_nodes = 10

# other settings
pop_size = 9
num_epoch = 3
mut_chance = 70

# global vars
best_genetics = (0, None)


hps_map = ('conv_layers', 'conv_filters', 'conv_kern', 'conv_activations', 'lin_layers', 
           'lin_widths', 'lin_activations', 'learn_rates', 'optimizers')

hps = {'conv_layers': [x for x in range(1, 3)],
       'conv_filters': [x for x in range(8, 64, 8)],
       'conv_kern': [x for x in range(1, 5)], 
       'conv_activations': ['linear', 'relu', 'elu', 'tanh', 'sigmoid'],

       'lin_layers': [x for x in range(1, 3)],
       'lin_widths': [x for x in range(32, 256, 32)],
       'lin_activations': ['linear', 'relu', 'elu', 'tanh', 'sigmoid'],

       'learn_rates': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
       'optimizers': ['sgd', 'rmsprop', 'adam', 'momentum', 'adagrad'],
       }


def breed(parent1, parent2):
    crossover = random.randint(1, len(parent2)-2)
    child = mutate(parent1[0:crossover] + parent2[crossover:])
    return child

def mutate(child):
    if random.randint(0, 100) <= mut_chance:
        gene = random.randint(0, len(hps_map)-1)
        gene_name = hps_map[gene]
        child[gene] = random.choice(hps[gene_name])
    return child

def make_model(key, in_layer, conv_layer, conv_filter, conv_kern, conv_activation, lin_layer, 
               lin_width, lin_activation, learn_rate, optimizer):
    network = in_layer
    # conv layers
    for i in range(conv_layer):
        network = tflearn.conv_2d(network, conv_filter, conv_kern, 
                                  activation=conv_activation, regularizer="L2", 
                                  name="conv_2d_{}_{}".format(key, i))

    # max pool layers
    network = tflearn.max_pool_2d(network, 2, name="max_pool_2d_{}".format(key))

    # linear layers
    for i in range(lin_layer):
        network = tflearn.fully_connected(network, lin_width, 
                                          activation=lin_activation,
                                          name="fully_connected_{}_{}".format(key, i))

    # output layer
    network = tflearn.fully_connected(network, out_nodes, activation='softmax', name='output_{}'.format(key))
    network = tflearn.regression(network, optimizer=optimizer, learning_rate=learn_rate,
                                 loss='categorical_crossentropy', name='target_{}'.format(key))
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

def train_network(key, genes, results, gpu_mem_fraction):
    print('network {}'.format(key))
    net_def = zip(hps_map, genes)
    for item in net_def:
        print(item)
    #tflearn.config.init_graph(gpu_memory_fraction=gpu_mem_fraction)
    in_layer = tflearn.input_data(shape=in_shape, name='input')
    net = make_model(key, in_layer, *genes)
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(X, Y, n_epoch=num_epoch, show_metric=False, 
              run_id='ga_convnet_mnist', batch_size=16)
    score = model.evaluate(Xval, Yval, batch_size=16)[0]
    results.append((score, genes))


def main():
    try:
        pop = init_pop()
        generation = 1
        while True:
            print('Training Generation {}'.format(generation))
            manager = mp.Manager()
            results = manager.list()
            #results = []
            nets = []
            gpu_mem_fraction = 0.98/3
            processes = []

            for key, genes in enumerate(pop):
                print('network {}'.format(key))
                net_def = zip(hps_map, genes)
                for item in net_def:
                    print(item)
                proc = mp.Process(target=train_network, args=(key, genes, results, gpu_mem_fraction))
                processes.append(proc)
                proc.start()
            
            # join all threads...
            for proc in processes:
                proc.join()

            # make results a local lists for sorting n shit
            res_list = []
            for item in results:
                res_list.append(item)
            results = res_list
            
            results.sort(key=lambda res: res[0], reverse=True)

            print(results[0][0])
            global best_genetics
            if results[0][0] > best_genetics[0]:
                best_genetics = results[0]

            top3 = results[0:3]
            print(top3)
            pop = handle_breed(top3)
            print_best()
            print('end of generation {}'.format(generation))
            generation += 1

    except KeyboardInterrupt:
        print_best()


if __name__ == '__main__':
    main()
