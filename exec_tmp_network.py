import tmp_network as network
from tmp_network import FullyConnectedLayer
import mnist_loader
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([FullyConnectedLayer(784, 30, network.sigmoid_func),
                       FullyConnectedLayer(30, 10, network.sigmoid_func)],
                      10, network.CrossEntropyCost)
#net.GD(training_data, 30, 1.0, test_data)
net.SGD(training_data, 10, 30, 1.0, test_data)
