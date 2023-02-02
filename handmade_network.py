class FullyConnectedLayer():
    def __init__(self, in, out, activation_func):
       self.in = in
       self.out = out
       self.activation_func = activation_func
       self.default_weight_initializer()

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    def feedforward(self, z):
        pass

class Network():
    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size

    def GD():
        pass
    def SGD():
        pass




class sigmoid_function():
    def output(z):
        pass
    def prime(z):
        pass
class tanh_function():
    def output(z):
        pass
    def prime(z):
        pass
