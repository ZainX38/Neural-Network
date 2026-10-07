import numpy as np
import pandas as pd

np.random.seed(0)   # Makes sure random values generated are always the same - for debugging purposes

LAYER_SIZES = [784, 16, 16, 10]
NUM_IMAGES_TO_PROCESS = 1000


def sigmoid_function(x):
    # Activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def cost_function(hot_encoded, final_out):
        # It uses the hot encoded label to see how bad the network performed
        # The greater the total sum, the worse it performed as values are more spread and further from target
        # High average cost = bad || Low average cost = good
        result_array = (hot_encoded - final_out)**2
        total_sum = np.sum(result_array)

        return total_sum

def network_accuracy(neural_network, image_pixels, hot_encoded, num_images):
    # Loops through num_images images and gives the accuracy - how many the network got correct
    correct_guess = 0

    for i in range(num_images):
        final_out = neural_network.feedforward(image_pixels[i])

        out_max_index = np.argmax(final_out) # Gets the index of the greatest number
        label_max_index = np.argmax(hot_encoded[i]) # Find the index of the 1, which is the result
        
        if out_max_index == label_max_index:
            correct_guess += 1

    accuracy = (correct_guess / num_images) * 100

    return accuracy

def average_cost(neural_network, image_pixels, hot_encoded, num_images):
    # Finds what the average cost is to see how the network is performing
    # High average cost = bad || Low average cost = good
    total_cost = 0

    for i in range(num_images):
        final_out = neural_network.feedforward(image_pixels[i])

        cost = cost_function(hot_encoded[i], final_out)
        total_cost += cost

    return (total_cost / num_images)


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
        total = sigmoid_function(total)
        
        return total

class NeuralNetwork:
    def __init__(self, layers_size):
        self.input_size = layers_size[0]    # 784 Inputs
        self.h1_size = layers_size[1]       # 16 neurons in 1st hidden layer
        self.h2_size = layers_size[2]       # 16 neurons in 2nd hidden layer
        self.out_size = layers_size[3]      # 10 neurons in output layer

        self.layers = []

        # Loop through layer_sizes, but start at index 1 (index 0 has the inputs)
        for i in range(1, len(layers_size)):
            num_neurons = layers_size[i]
            num_inputs = layers_size[i-1]

            # Create the layer with all the neurons
            layer = [Neuron(num_inputs) for _ in range(num_neurons)]

            # Populates layers array with all the layers
            self.layers.append(layer)

    def feedforward(self, x):
        current_inputs = x

        # Get a layer
        # Loop through this layer and feedforward every single element
        # Get the output
        # Pass this output as the input of the next layer
        # Return this same variable as the final output

        for layer in self.layers:
            '''
            out = []
            
                        for neuron in layer:
                            out.append(neuron.feedforward(current_inputs))
                        current_inputs = out
            '''

            # Current inputs processed and return an output which gets passed as the output for the next layer
            # This is the same as the bracketed code
            current_inputs = [neuron.feedforward(current_inputs) for neuron in layer]

        # The last array to come out from the array is the final output
        final_out = np.array(current_inputs)

        return final_out


def get_mnist_data():
    mnist_data = pd.read_csv("mnist_train.csv", header=None)
    mnist_data = np.array(mnist_data)

    labels = mnist_data[0:, 0]  # Returns first column which has the labels (results)

    # 60000 rows (training), and 10 columns
    hot_encoded = np.zeros((len(labels), 10))

    # Hot encode labels into hot_encode
    for i in range(len(labels)):
        hot_encoded[i][labels[i]] = 1
        

    pixel_data = mnist_data[0:, 1:] # Returns all rows and columns (except 1st column - labels). These are pixel values from 0-255
    pixel_data = pixel_data / 255   # Scale pixel values from 0-1 - these are the input values for the Neural Network
    # print(pixel_data.dtype, pixel_data.min(), pixel_data.max())

    return pixel_data, hot_encoded


if __name__ == "__main__":
    image_pixels, hot_encoded = get_mnist_data()

    neural_network = NeuralNetwork(LAYER_SIZES)
    final_out = neural_network.feedforward(image_pixels[0])
    print(f"FINAL: {final_out}")

    cost_error = cost_function(hot_encoded[0], final_out)
    print(f"Cost Error: {cost_error:.2f}")

    accuracy = network_accuracy(neural_network, image_pixels, hot_encoded, NUM_IMAGES_TO_PROCESS)
    print(f"Accuracy: {accuracy}%")

    avg_cost = average_cost(neural_network, image_pixels, hot_encoded, NUM_IMAGES_TO_PROCESS)
    print(f"Average Cost: {avg_cost:.2f}")

    


