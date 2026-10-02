import numpy as np
import pandas as pd

np.random.seed(0)   # Makes sure random values generated are always the same - for debugging purposes

def sigmoid_function(x):
    # Activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self):
        # Creates random weights for each neuron
        # The weights are an array of 784 random number from a normal distribution from -2 to 2
        self.weights = np.random.randn(784)
        self.bias = 0

    def feedforward(self, input):
        # Feedforward is the dot product of the inputs and the weights plus the bias, and with the activation function
        # f[(a1w1 + a2w2 + anwn) + b] --> f[a•w + b]
        total = np.dot(input, self.weights) + self.bias
        return sigmoid_function(total)

class NeuralNetwork:
    def __init__(self, neurons):

        # Creates the hidden layer
        self.neurons = neurons  # List of all 16 neurons

    def feedforward(self, x):
        pass


mnist_data = pd.read_csv("mnist_train.csv", header=None)
mnist_data = np.array(mnist_data)

labels = mnist_data[0:, 0]  # Returns first column which has the labels (results)

pixel_data = mnist_data[0:, 1:] # Returns all rows and columns (except 1st column - labels). These are pixel values from 0-255
pixel_data = pixel_data / 255   # Scale pixel values from 0-1 - these are the input values for the Neural Network
# print(pixel_data.dtype, pixel_data.min(), pixel_data.max())

HIDDEN_LAYER_SIZE = 16
image_pixels = pixel_data[0]    # INPUT LAYER - Gets the first row which is are the pixels of the 1st image
neuron_container = []

# Creates the neurons for the 1st Hidden Layer
for i in range(HIDDEN_LAYER_SIZE):
    neuron = Neuron()
    print(neuron.feedforward(image_pixels))
    neuron_container.append(neuron)

    


