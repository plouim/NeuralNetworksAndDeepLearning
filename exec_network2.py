import network2 as network
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])
net.SGD(training_data, 10, 10, 1.0, 0.0, evaluation_data=validation_data,
        monitor_training_cost=False,
        monitor_training_accuracy=True,
        monitor_evaluation_cost=False,
        monitor_evaluation_accuracy=True)
