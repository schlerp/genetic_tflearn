import tensorflow as tf
import tflearn
from tflearn.data_utils import *
from multiprocessing import Process, Manager


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
num_epoch = 10


def train_net1(res):
    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.45
    #session = tf.Session(config=config)    
    #with tf.Graph().as_default():
    tflearn.config.init_graph(gpu_memory_fraction=0.45)
    net1 = tflearn.input_data([None, 28, 28, 1], name='input1')
    net1 = tflearn.fully_connected(net1, 128)
    net1 = tflearn.fully_connected(net1, 128)
    net1 = tflearn.fully_connected(net1, 10, activation='softmax')
    net1 = tflearn.regression(net1, loss='categorical_crossentropy', name='target1')
    model1 = tflearn.DNN(net1)
    
    model1.fit(X, Y, n_epoch=num_epoch, show_metric=False, 
                              run_id='ga_convnet_mnist')
    score = model1.evaluate(Xval, Yval)[0]
    res.append((score, ('example1', 'genes1', 1, 2, 3)))


def train_net2(res):
    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.45
    #session = tf.Session(config=config)
    #with tf.Graph().as_default():
    tflearn.config.init_graph(gpu_memory_fraction=0.45)
    net2 = tflearn.input_data([None, 28, 28, 1], name='input2')
    net2 = tflearn.fully_connected(net2, 128)
    net2 = tflearn.fully_connected(net2, 128)
    net2 = tflearn.fully_connected(net2, 10, activation='softmax')
    net2 = tflearn.regression(net2, loss='categorical_crossentropy', name='target2')
    model2 = tflearn.DNN(net2)
    
    model2.fit(X, Y, n_epoch=num_epoch, show_metric=False, 
                              run_id='ga_convnet_mnist')
    score = model2.evaluate(Xval, Yval)[0]
    res.append((score, ('example2', 'genes2', 1, 2, 3)))


#net1 = tflearn.input_data(in_shape, name='input1')
#net1 = tflearn.fully_connected(net1, 128)
#net1 = tflearn.fully_connected(net1, 128)
#net1 = tflearn.fully_connected(net1, out_nodes, activation='softmax')
#net1 = tflearn.regression(net1, loss='categorical_crossentropy', name='target1')
#model1 = tflearn.DNN(net1)

#net2 = tflearn.input_data(in_shape, name='input2')
#net2 = tflearn.fully_connected(net2, 128)
#net2 = tflearn.fully_connected(net2, 128)
#net2 = tflearn.fully_connected(net2, out_nodes, activation='softmax')
#net2 = tflearn.regression(net2, loss='categorical_crossentropy', name='target2')
#model2 = tflearn.DNN(net2)

manager = Manager()
results = manager.list()

process1 = Process(target=train_net1, args=(results,))
process2 = Process(target=train_net2, args=(results,))
process1.start()
process2.start()
process1.join()
process2.join()

for item in results:
    print(item)
    
print('Done!')