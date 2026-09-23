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


weights = np.array([0, 1])      # w1 = 0, w2 = 1
bias = 4
n = Neuron(weights, bias)

inputs = np.array([2, 3])       # a1 = 2, a2 = 3
print(n.feedforward(inputs))    # 0.9990889488055994