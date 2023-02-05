import handmade_network as network
from handmade_network import FullyConnectedLayer
import mnist_loader
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([FullyConnectedLayer(784, 30, network.sigmoid_func),
                       FullyConnectedLayer(30, 10, network.sigmoid_func)],
                      10, network.QuadraticCost)
net.GD(training_data[0:10000], 10, 5.0, test_data)
