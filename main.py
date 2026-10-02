import numpy as np
import pandas as pd

np.random.seed(0)   # Makes sure random values generated are always the same - for debugging purposes

def sigmoid_function(x):
    # Activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, num_inputs):
        self.bias = 0

        # Weights are scaled so they stay in the responsive range of sigmoid's function
        scale_factor = 1/np.sqrt(num_inputs)
        weights_random = np.random.randn(num_inputs)
        self.weights = weights_random * scale_factor

    def feedforward(self, inputs):
        # Feedforward is the dot product of the inputs and the weights plus the bias, and with the activation function
        # f[(a1w1 + a2w2 + anwn) + b] --> f[a•w + b]
        total = np.dot(inputs, self.weights) + self.bias
        return sigmoid_function(total)

class NeuralNetwork:
    def __init__(self, layers_size):
        self.input_size = layers_size[0]    # 784 Inputs
        self.h1_size = layers_size[1]       # 16 neurons in 1st hidden layer
        self.h2_size = layers_size[2]       # 16 neurons in 2nd hidden layer
        self.out_size = layers_size[3]      # 10 neurons in output layer

        # Creates the 1st Hidden Layer, which has 16 neurons
        self.h1_neurons = []
        for i in range(self.h1_size):
            # Weights are passed using input_size as this corresponds to weight size. Weight size = input size
            neuron = Neuron(self.input_size)
            self.h1_neurons.append(neuron)

        # Creates 2nd Hidden Layer, which has 16 neurons
        self.h2_neurons = []
        for i in range(self.h2_size):
            neuron = Neuron(self.h1_size)
            self.h2_neurons.append(neuron)

        # Creates Output Layer, which has 10 neurons
        self.out_neurons = []
        for i in range(self.out_size):
            neuron = Neuron(self.h2_size)
            self.out_neurons.append(neuron)

    def feedforward(self, x):
        h1_out = []
        for neuron in self.h1_neurons:
            out = neuron.feedforward(x)
            h1_out.append(out)

        h2_out = []
        for neuron in self.h2_neurons:
            out = neuron.feedforward(h1_out)
            h2_out.append(out)

        final_out = []
        for neuron in self.out_neurons:
            out = neuron.feedforward(h2_out)
            final_out.append(out)

        return final_out




mnist_data = pd.read_csv("mnist_train.csv", header=None)
mnist_data = np.array(mnist_data)

labels = mnist_data[0:, 0]  # Returns first column which has the labels (results)

pixel_data = mnist_data[0:, 1:] # Returns all rows and columns (except 1st column - labels). These are pixel values from 0-255
pixel_data = pixel_data / 255   # Scale pixel values from 0-1 - these are the input values for the Neural Network
# print(pixel_data.dtype, pixel_data.min(), pixel_data.max())

LAYER_SIZES = [784, 16, 16, 10]
image_pixels = pixel_data[0]    # INPUT LAYER - Gets the first row which is are the pixels of the 1st image

neural_network = NeuralNetwork(LAYER_SIZES)
print(neural_network.feedforward(image_pixels))

    


