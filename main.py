import numpy as np
import pandas as pd

np.random.seed(0)   # Makes sure random values generated are always the same - for debugging purposes

def sigmoid_function(x):
    # Activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, weights):
        # Creates random weights for each neuron
        # The weights are an array of 784 random number from a normal distribution from -2 to 2
        self.weights = np.random.randn(weights)
        self.bias = 0

    def feedforward(self, input):
        # Feedforward is the dot product of the inputs and the weights plus the bias, and with the activation function
        # f[(a1w1 + a2w2 + anwn) + b] --> f[a•w + b]
        total = np.dot(input, self.weights) + self.bias
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
            neurons = Neuron(self.input_size)
            self.h1_neurons.append(neurons)

        # Creates 2nd Hidden Layer, which has 16 neurons
        self.h2_neurons = []
        for i in range(self.h2_size):
            neurons = Neuron(self.h1_size)
            self.h2_neurons.append(neurons)

        # Creates Output Layer, which has 10 neurons
        self.final_out = []
        for i in range(self.out_size):
            neurons = Neuron(self.h2_size)
            self.final_out.append(neurons)

    def feedforward(self, x):
        pass



mnist_data = pd.read_csv("mnist_train.csv", header=None)
mnist_data = np.array(mnist_data)

labels = mnist_data[0:, 0]  # Returns first column which has the labels (results)

pixel_data = mnist_data[0:, 1:] # Returns all rows and columns (except 1st column - labels). These are pixel values from 0-255
pixel_data = pixel_data / 255   # Scale pixel values from 0-1 - these are the input values for the Neural Network
# print(pixel_data.dtype, pixel_data.min(), pixel_data.max())

LAYERS_SIZE = [784, 16, 16, 10]
image_pixels = pixel_data[0]    # INPUT LAYER - Gets the first row which is are the pixels of the 1st image

neural_network = NeuralNetwork(LAYERS_SIZE)
print(neural_network.feedforward(image_pixels))

    


