import numpy as np
from numpy import random

class FullyConnectedLayer():
    def __init__(self, n_in, n_out, activation_func):
       self.n_in = n_in
       self.n_out = n_out
       self.activation_func= activation_func
       self.default_weight_initializer()

    def default_weight_initializer(self):
        self.biases = np.random.randn(self.n_out, 1) 
        self.weights = np.random.randn(self.n_out, self.n_in)/np.sqrt(self.n_in)

    def large_weight_initializer(self):
        self.biases = np.random.randn(self.n_out, 1)
        self.weights = np.random.randn(self.n_out, self.n_in)

    def feedforward(self, x):
        return self.activation_func.output(self.weights.dot(x) + self.biases)
    
class Network():
    def __init__(self, layers, mini_batch_size, cost_func):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.cost_func = cost_func

    def feedforward(self, x):
        for l in self.layers:
            x = l.feedforward(x)
        return x

    def backprop(self, x, y):
        activation = x
        activations = [x]
        zs = []
        nabla_w = [np.zeros(l.weights.shape) for l in self.layers]
        nabla_b = [np.zeros(l.biases.shape) for l in self.layers]

        for layer in self.layers:
            z = layer.weights.dot(activation) + layer.biases
            zs.append(z)
            activation = layer.activation_func.output(z)
            activations.append(activation)
        delta = self.cost_func.delta(activations[-1], y) \
                * self.layers[-1].activation_func.prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = delta.dot(activations[-2].T)

        for l in range(2, len(self.layers)+1):
            delta = self.layers[-l+1].weights.T.dot(delta) \
                    * self.layers[-l].activation_func.prime(zs[-l])
            nabla_w[-l] = delta.dot(activations[-l-1].T)
            nabla_b[-l] = delta
        return nabla_b, nabla_w

    def GD(self, training_data, epochs, eta, evaluation_data):
        for i in range(epochs):
            nabla_w = [np.zeros(l.weights.shape) for l in self.layers]
            nabla_b = [np.zeros(l.biases.shape) for l in self.layers]

            for x, y in training_data:
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

            for l, nw, nb in zip(self.layers, nabla_w, nabla_b):
                l.weights -= eta/len(training_data)*nw
                l.biases -= eta/len(training_data)*nb
                
            # display result
            print("epoch: {}".format(i))
            n = len(training_data)
            n_data = len(evaluation_data)
            training_accuracy, evaluation_accuracy = [], []
            # training accuracy
            accuracy = self.accuracy(training_data, convert=True)
            training_accuracy.append(accuracy)
            print("Accuracy on training data: {} / {}".format(accuracy, n))
            # test accuracy
            accuracy = self.accuracy(evaluation_data)
            evaluation_accuracy.append(accuracy)
            print("Accuracy on evaluation data: {} / {}"
                   .format(self.accuracy(evaluation_data), n_data))
            print("")

    
    def SGD(self, training_data, mini_batch_size, epochs, eta, evaluation_data):
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, len(training_data), mini_batch_size)]

            for mini_batch in mini_batches:
                nabla_w = [np.zeros(l.weights.shape) for l in self.layers]
                nabla_b = [np.zeros(l.biases.shape) for l in self.layers]
                for x, y in mini_batch:
                    delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                    nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                    nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

                for l, nw, nb in zip(self.layers, nabla_w, nabla_b):
                    l.weights -= eta/mini_batch_size*nw
                    l.biases -= eta/mini_batch_size*nb
               
            # display result
            print("epoch: {}".format(i))
            n = len(training_data)
            n_data = len(evaluation_data)
            training_accuracy, evaluation_accuracy = [], []
            # training accuracy
            accuracy = self.accuracy(training_data, convert=True)
            training_accuracy.append(accuracy)
            print("Accuracy on training data: {} / {}".format(accuracy, n))
            # test accuracy
            accuracy = self.accuracy(evaluation_data)
            evaluation_accuracy.append(accuracy)
            print("Accuracy on evaluation data: {} / {}"
                   .format(self.accuracy(evaluation_data), n_data))
            print("")

    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)


class sigmoid_func:
    @staticmethod
    def output(z):
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def prime(z):
        return  np.exp(-z) / ((1 + np.exp(-z))**2 )

class tanh_func:
    @staticmethod
    def output(z):
        return (np.exp(z)-np.exp(-z)) / (np.exp(z)+np.exp(-z))

    @staticmethod
    def prime(z):
        return 1 - ((np.exp(z)-np.exp(-z)) / (np.exp(z)+np.exp(-z)))**2

class QuadraticCost:
    @staticmethod
    def output(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(a,y):
        return (a-y)
    #@staticmethod
    #def delta(a, y, z, activation_func):
    #    return (a-y) * activation_func.prime(z)

class CrossEntropyCost(object):

    @staticmethod
    def output(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(a, y):
        return (a-y)

