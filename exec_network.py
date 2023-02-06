import network
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])
net.SGD(training_data, 50000, 10, 1.0, test_data=test_data)
