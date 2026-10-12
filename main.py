import numpy as np
import pandas as pd

np.random.seed(0)   # Makes sure random values generated are always the same - for debugging purposes

LAYER_SIZES = [784, 16, 16, 10]

# Hyperparameter. Used to calculate the new weight.
# A small value such as 0.01 makes learning slow as it's moving only a tiny bit in the graph to the minima
# A large value such as 10 makes learning impossible as it can never find a minima cause it overshoots
LEARNING_RATE = 0.1
EPOCHS = 3
NUM_IMAGES_TO_PROCESS = 10000


# ============================= HELPER FUNCITIONS ====================================
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

def get_mnist_data(file_path):
    mnist_data = pd.read_csv(file_path, header=None)
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

# =================================== NEURON CLASS ======================================
class Neuron:
    def __init__(self, num_inputs):
        self.bias = 0

        # Weights are scaled so they stay in the responsive range of sigmoid's function
        scale_factor = 1/np.sqrt(num_inputs)
        weights_random = np.random.randn(num_inputs)
        self.weight = weights_random * scale_factor

        # Each neuron will store its input and output, so it can be seen for backpropagation
        self.input = None
        self.output = None
        self.delta = 0

    def feedforward(self, inputs):
        # Feedforward is the dot product of the inputs and the weights plus the bias, and with the activation function
        # f[(a1w1 + a2w2 + anwn) + b] --> f[a•w + b]
        total = np.dot(inputs, self.weight) + self.bias
        out = sigmoid_function(total)

        self.input = inputs
        self.output = out
        
        return out


# ================================ NEURAL NETWORK CLASS ================================
class NeuralNetwork:
    def __init__(self, layers_size):
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
            # Current inputs processed and return an output which gets passed as the output for the next layer
            # This is the same as the bracketed code
            current_inputs = [neuron.feedforward(current_inputs) for neuron in layer]

        # The last array to come out from the array is the final output
        final_out = np.array(current_inputs)

        return final_out

    def backpropagation(self, inputs, hot_encoded, learning_rate):

        # Runs feedforward at start, so every neuron stores its input and output
        self.feedforward(inputs)

        '''
        1. Work out delta for output layer
            - Find a (final output from output layer)
            - Find y (expected output, which is hot encoded)
            - Calculate delta using: a*(1-a) * 2*(a-y)

        2. Work out delta for hidden layers:
            - Get the weight of neuron in output layer
            - Multiply weight * previously calculated delta
            - Calculate sum of this: sum(weight * delta)
            - Calculate sigmoid derivative, a(1-a) (where a is the output of the hidden neuron)
            - Delta for hidden layer is a(1-a) * sum(weight * delta)

        3. Calculate new weights:
            - Work out gradient: dC/dw = input * delta
            - Work out: Learning rate * gradient
            - New weight = Old weight - learning rate * gradient
        '''

        for j, neuron in enumerate(self.layers[-1]):
            a = neuron.output
            y = hot_encoded[j]

            neuron.delta = a*(1-a) * 2*(a-y)


        # Start at first hidden layer (layers - 2) and moved backwards to 1st hidden layer at steps of -1
        for i in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[i]  # Hidden layer whose deltas are being calculated
            next_layer = self.layers[i+1]   # It is the layer after, which has the neurons with already calculated deltas



            for j, neuron in enumerate(current_layer):
                blame = 0   # This accumulates and stores the sum: sum(weight * delta)

                for next_neuron in next_layer:
                    # next_neuron.weight[j] is the weight connecting the current neuron at index j to the next neuron
                    blame += next_neuron.weight[j] * next_neuron.delta

                a = neuron.output
                sigmoid_derivative = a*(1-a)

                neuron.delta = sigmoid_derivative * blame


        for layer in self.layers:
            for neuron in layer:
                # Calculate gradient
                gradient = np.array(neuron.input) * neuron.delta

                neuron.weight-= learning_rate * gradient
                neuron.bias-= learning_rate * neuron.delta

# ================================= MEAURE NETWORK PERFORMANCE ==========================================

# Calculates accuracy - how many images it got right
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

# Finds what the average cost is to see how the network is performing
# High average cost = bad || Low average cost = good
def average_cost(neural_network, image_pixels, hot_encoded, num_images):
    total_cost = 0

    for i in range(num_images):
        final_out = neural_network.feedforward(image_pixels[i])

        cost = cost_function(hot_encoded[i], final_out)
        total_cost += cost

    return (total_cost / num_images)


# ========================================== MAIN PROGRAM ============================================

if __name__ == "__main__":
    image_pixels, hot_encoded = get_mnist_data("mnist_train.csv")

    training_pixels = image_pixels[:NUM_IMAGES_TO_PROCESS]
    training_labels = hot_encoded[:NUM_IMAGES_TO_PROCESS]

    test_pixels = image_pixels[50000:]
    test_labels = hot_encoded[50000:]

    neural_network = NeuralNetwork(LAYER_SIZES)

    for epoch in range(EPOCHS):
        # Shuffles indices, so pixels and labels stay matched
        # This shuffles the data, which improves the network's ability to learn
        order = np.random.permutation(NUM_IMAGES_TO_PROCESS)

        for i in order:
            neural_network.backpropagation(training_pixels[i], training_labels[i], LEARNING_RATE)
        
        accuracy = network_accuracy(neural_network, test_pixels, test_labels, len(test_pixels))
        avg_cost = average_cost(neural_network, test_pixels, test_labels, len(test_pixels))

        print(f"EPOCH {epoch + 1}:")
        print(f"Accuracy: {accuracy}%")
        print(f"Average Cost: {avg_cost}")