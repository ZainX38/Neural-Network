import numpy as np

def sigmoid_function(x):
    # Activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, input):
        # Feedforward is the dot product of the inputs and the weights plus the bias, and with the activation function
        # f[(a1w1 + a2w2 + anwn) + b] --> f[a•w + b]
        total = np.dot(input, self.weights) + self.bias
        return sigmoid_function(total)

class NeuralNetwork:
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        # Creates the hidden layer - it has only 2 neurons at the moment
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)

        # This is the output after the hidden layer
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        # Finds the output of the neurons in the hidden layer
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        # Inputs of o1 are the outputs of h1 and h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1



network = NeuralNetwork()
x = np.array([2, 3])

print(network.feedforward(x))   # 0.7216325609518421