# Handwritten Digit Recognition from Scratch

A neural network that learns to read handwritten digits (0–9), built using only Python, NumPy and pandas (just for reading the CSV file). No machine learning libraries were used, so every single part from neurons to backpropagation is built from scratch.

## Contents

1. [The problem and the dataset](#1-the-problem-and-the-dataset)
2. [Loading the data](#2-loading-the-data)
3. [The neuron](#3-the-neuron)
4. [Feedforward](#4-feedforward)
5. [The network design](#5-the-network-design)
6. [The forward pass](#6-the-forward-pass)
7. [Measuring how wrong the network is](#7-measuring-how-wrong-the-network-is)
8. [How the network learns: gradient descent](#8-how-the-network-learns-gradient-descent)
9. [Backpropagation](#9-backpropagation)
10. [The learning rate](#10-the-learning-rate)
11. [Running the code](#12-running-the-code)

---

## 1. The problem and the dataset

**Goal:** given a 28×28 pixel image of a handwritten digit, predict which digit it is.

A normal coding approach, e.g. "if the top is curved, it's a 9", doesn't work, because everyone writes numbers differently. Instead, a neural network **learns** the patterns from thousands of labelled examples. It starts by guessing randomly and gradually adjusts itself to make fewer mistakes.

The [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database):

| File | Images | Used for |
| --- | --- | --- |
| `mnist_train.csv` | 60,000 | Training |
| `mnist_test.csv` | 10,000 | Testing on images never seen during training |

**Structure of each file:**

- Each **row** is one image.
- **Column 0** is the label: the correct digit (0–9).
- **Columns 1–784** are the pixels. The 28 × 28 grid is flattened into one row of 784 values.
- Each pixel is a whole number from **0** (white) to **255** (black).
- There is **no header row**

The data files aren't in this repository because they exceed GitHub's 100 MB file limit (they're listed in `.gitignore`). Download them separately into the project folder.

## 2. Loading the data

```python
def get_mnist_data():
    mnist_data = pd.read_csv("mnist_train.csv", header=None)
    mnist_data = np.array(mnist_data)

    labels = mnist_data[0:, 0]      # Every row, column 0

    pixel_data = mnist_data[0:, 1:] # Every row, every column except column 0
    pixel_data = pixel_data / 255   # Scale pixel values from 0-1

    return pixel_data, labels
```

**`header=None`:** by default, pandas treats the first line as column names. Since the first line is an image, it would be lost, leaving 59,999 images instead of 60,000.

**Slicing:**
- **Labels:** extract from column 0 (in every single row)
- **Pixel data:** extract from column 1 to the end (column 784)

**Dividing by 255:**
Scales every pixel to between 0 and 1. The reason becomes clear in [Section 3](#why-scale-them-saturation) (large inputs make neurons stop learning)

**Result:**
`pixel_data` has shape `(60000, 784)`. One image is one row, for example `pixel_data[0]`.

## 3. The neuron

The neuron is where the actual thinking happens. It must have 2 properties: **weight** and **bias**.

```python
np.random.seed(0)   # Makes sure random values generated are always the same - for debugging purposes

class Neuron:
    def __init__(self, num_inputs):
        self.bias = 0

        # Weights are scaled so they stay in the responsive range of sigmoid's function
        scale_factor = 1/np.sqrt(num_inputs)
        weights_random = np.random.randn(num_inputs)
        self.weight = weights_random * scale_factor
```

Each neuron gets **one weight per input**, so `num_inputs` decides how many weights it has.

### Why random

If every neuron started with the same weights, they would all calculate exactly the same output, and later receive exactly the same updates. They would stay identical forever, so 16 neurons would be no more useful than one. Random starting weights make every neuron different, so each can learn to detect something different. This is called **symmetry breaking**.

`np.random.randn` draws numbers from a normal distribution: mostly between −3 and 3, centred on 0. Each neuron makes its **own** random array. Biases start at 0, since the random weights already break the symmetry.

### Why scale them: saturation

The neuron's output goes through the sigmoid function (explained in [Section 4](#the-sigmoid-activation-function)), which squashes it between 0 and 1.

Without scaling, a neuron in the first hidden layer adds up 784 `pixel × weight` values, with weights around ±1. The total is often large, like ±10. With unscaled weights, the 16 neurons of the first hidden layer output values like:

```
0.0000653   0.9999779   0.0000000261   0.9999976   0.0000000016   ...
```

Most are pinned near 0 or 1. Far from zero, the sigmoid curve is almost flat:

| z | Slope of sigmoid |
| --- | --- |
| 0 | 0.25 (the steepest) |
| 5 | 0.0066 |
| 12 | 0.000006 |

When the curve is flat, changing a weight barely changes the output, so the neuron can hardly learn. This is called **saturation**. It's also why the pixels are divided by 255: inputs up to 255 would make it far worse.

**The fix (Xavier initialisation):** more inputs means a bigger sum, so the weights are made smaller in proportion, multiplied by $1/\sqrt{\text{number of inputs}}$. For the first hidden layer, $\sqrt{784} = 28$, so the weights end up around ±0.1, and $z$ stays roughly between −1 and 1, in sigmoid's steep, responsive middle.

It's the square root of the **number of inputs**, not of the weights. Negative weights stay negative, since multiplying by a positive number keeps the sign.

### Why `np.random.seed(0)`

It makes the "random" numbers the same on every run, so results are reproducible. When something goes wrong, exact numbers can be compared between runs to track it down.

## 4. Feedforward

With the weight and bias, and the pixel data (inputs, which are 784 pixels), we can now calculate the output of a neuron:

```python
class Neuron:
    def feedforward(self, inputs):
        # f[(a1w1 + a2w2 + anwn) + b] --> f[a•w + b]
        total = np.dot(inputs, self.weight) + self.bias
        out = sigmoid_function(total)

        return out
```

This function does three things:

1. **Multiplies each input by a weight.** A weight says how much the neuron cares about that input. Positive pushes the neuron up, negative pushes it down, near zero ignores it.
2. **Adds them up, plus a bias.** The bias shifts how easily the neuron activates, whatever the inputs are.
3. **Squashes the result** through an activation function into the range 0 to 1.

$$
z = w_1 a_1 + w_2 a_2 + \dots + w_n a_n + b = \mathbf{w} \cdot \mathbf{a} + b
$$

$$
\text{output} = \sigma(z)
$$

`np.dot` does the multiplying and adding in one fast operation.

**Example:** with inputs `[2, 3]`, weights `[0, 1]` and bias `4`: $z = (0 \times 2) + (1 \times 3) + 4 = 7$, and the output is $\sigma(7) = 0.999$.

### The sigmoid activation function

```python
def sigmoid_function(x):
    # Activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))
```

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

| z | σ(z) |
| --- | --- |
| −10 | ≈ 0.00005 |
| −2 | 0.12 |
| 0 | 0.5 |
| 2 | 0.88 |
| 10 | ≈ 0.99995 |

**Why an activation function is needed:** without one, a neuron is just multiplication and addition. Stacking layers of multiplication and addition still only gives multiplication and addition, so the whole network could only learn straight-line relationships. The bend in the sigmoid curve is what lets it learn complex patterns, like the shapes of digits.

## 5. The network design

Now we have designed a neuron, which can calculate an output using inputs. However, one neuron alone isn't enough to recognise a digit. So neurons are arranged in **layers**, where the outputs of one layer become the inputs of the next.

```python
LAYER_SIZES = [784, 16, 16, 10]
```

```
Input layer      Hidden layer 1    Hidden layer 2    Output layer
784 pixels  -->  16 neurons   -->  16 neurons   -->  10 neurons
                                                    (one per digit)
```

- **Input layer (784):** all pixel values of an image (not neurons)
- **Hidden layers (16, 16):** called "hidden" because we never see or set their outputs directly
- **Output layer (10):** output neuron *i* says how strongly the network thinks the image is digit *i*. The prediction is whichever output is largest.

Every neuron connects to every neuron in the previous layer. So each neuron needs **one weight per neuron in the layer before it**: 784 weights in hidden layer 1, but only 16 in hidden layer 2 and the output layer.

**Number of parameters:**

| Layer | Weights | Biases |
| --- | --- | --- |
| Hidden 1 | 784 × 16 = 12,544 | 16 |
| Hidden 2 | 16 × 16 = 256 | 16 |
| Output | 16 × 10 = 160 | 10 |
| **Total** | **12,960** | **42** |

The network has **13,002** numbers to learn.

### Building the layers

```python
class NeuralNetwork:
    def __init__(self, layers_size):
        self.layers = []

        # Loop through layer_sizes, but start at index 1 (index 0 has the inputs)
        for i in range(1, len(layers_size)):
            num_neurons = layers_size[i]
            num_inputs = layers_size[i-1]

            # Create the layer with all the neurons
            layer = [Neuron(num_inputs) for _ in range(num_neurons)]

            self.layers.append(layer)
```

| `i` | Neurons (`layers_size[i]`) | Inputs each (`layers_size[i-1]`) |
| --- | --- | --- |
| 1 | 16 | 784 |
| 2 | 16 | 16 |
| 3 | 10 | 16 |

- The loop **starts at 1**, because position 0 is the input layer, which has no neurons.
- `self.layers` is a **list of lists**: each inner list is one layer. `self.layers[0]` is hidden layer 1, and `self.layers[-1]` is the output layer.
- `append` adds each layer as **one item**, keeping the layers separate.
- **One loop instead of naming each layer separately** means the shape can be changed just by editing `LAYER_SIZES`. It also makes it easy to loop through the layers backwards, which learning will need ([Section 9](#9-backpropagation)).
- **The neurons are created once, in `__init__`,** and kept. Their weights are what the network learns, so they can't be recreated every time an image is processed.

Now that the layers are built, we can pass an image through the whole network:

## 6. The forward pass

To make a prediction, an image flows through the network from the inputs to the outputs:

```python
def feedforward(self, x):
    current_inputs = x

    for layer in self.layers:
        # Every neuron in the layer processes the current inputs.
        # Their outputs become the inputs for the next layer
        current_inputs = [neuron.feedforward(current_inputs) for neuron in layer]

    final_out = np.array(current_inputs)

    return final_out
```

1. `current_inputs` starts as the 784 pixels of one image.
2. Every neuron in the layer processes them. Their outputs are collected.
3. Those outputs become `current_inputs` for the next layer.
4. After the last layer, `current_inputs` holds the 10 outputs.

The outputs are returned as a NumPy array, so maths like subtraction works on all 10 at once (needed in the next section).

**Final Output Example:**
Before any learning, the outputs for the first image (a 5) are meaningless:

```
[0.527, 0.424, 0.634, 0.604, 0.753, 0.665, 0.366, 0.396, 0.681, 0.622]
```

The largest is at index 4, so the network guesses 4. Wrong, as expected from random weights. To deal with this, we firstly need to measure how wrong the network is:

## 7. Measuring how wrong the network is

To improve the network, there first needs to be a way of measuring how good or bad it is.

### One-hot encoding the labels

The network gives 10 outputs, but a label is a single number like `5`. These can't be compared directly. So each label is turned into 10 targets, where the correct digit's neuron should output 1, and all the others 0.

```
5  -->  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
         0  1  2  3  4  5  6  7  8  9
```

This is added to `get_mnist_data()`:

```python
# 60000 rows (training), and 10 columns
hot_encoded = np.zeros((len(labels), 10))

for i in range(len(labels)):
    hot_encoded[i][labels[i]] = 1

...
return pixel_data, hot_encoded
```

`np.zeros((len(labels), 10))` creates a 60,000 × 10 grid of zeros. For each image, a 1 is placed in the column matching its label. The result has shape `(60000, 10)`.

### The cost function

```python
def cost_function(hot_encoded, final_out):
    result_array = (hot_encoded - final_out)**2
    total_sum = np.sum(result_array)

    return total_sum
```

$$
C = \sum_{j=0}^{9} (a_j - y_j)^2
$$

where $a_j$ is output neuron $j$'s output and $y_j$ is its hot encoded target (0 or 1).

- **Subtract:** how far each output is from its target.
- **Square:** makes every error positive, so errors can't cancel each other out, and punishes big errors much more than small ones.
- **Sum:** one number for how wrong the network was on this image. 0 is perfect.

**The first image's cost before learning: ≈ 3.04.**

The cost is the most important number in the project, because **learning means making the cost smaller**.

### Average cost

One image can be lucky or unlucky. Averaging over many shows how the network does overall:

```python
def average_cost(neural_network, image_pixels, hot_encoded, num_images):
    total_cost = 0

    for i in range(num_images):
        final_out = neural_network.feedforward(image_pixels[i])

        cost = cost_function(hot_encoded[i], final_out)
        total_cost += cost

    return (total_cost / num_images)
```

Each image uses `[i]` for both its pixels and its label, so it's compared with its own answer. **Before learning: ≈ 3.25** over the first 1,000 images.

### Accuracy

Cost is hard for a person to interpret, so accuracy gives a readable measure: the percentage of images guessed correctly.

```python
def network_accuracy(neural_network, image_pixels, hot_encoded, num_images):
    correct_guess = 0

    for i in range(num_images):
        final_out = neural_network.feedforward(image_pixels[i])

        out_max_index = np.argmax(final_out)            # The network's guess
        label_max_index = np.argmax(hot_encoded[i])     # The correct answer

        if out_max_index == label_max_index:
            correct_guess += 1

    accuracy = (correct_guess / num_images) * 100

    return accuracy
```

`np.argmax` gives the **index** of the largest value
- For the outputs, that's the guess
- For the one-hot label, it's the position of the 1, which is the correct digit

**Before learning: 10.5%**, about the same as guessing one digit in 10 at random.

**Why have both?** Accuracy only says right or wrong. Cost is what the network actually learns from, because it changes smoothly: it shows *how* wrong, and in which direction to improve. Outputs of 0.51 vs 0.50 count as "correct", but the cost still shows there's a lot of room to improve.

Now that we can measure how wrong the network is, we need a way to make it less wrong:

## 8. How the network learns: gradient descent

The network has 13,002 weights and biases. **These are the only things it ever changes.** Everything else (`z`, the outputs, the cost) is calculated from them, and the labels are fixed. Learning means finding values for weights and biases that make the cost small.

### The Main Idea

Imagine the cost as a landscape of hills and valleys, where your position is set by the weights. Learning means walking **downhill** to the lowest point you can find. This lowest point is the minima in the graph.

For each weight, the question is: **"If I nudge this weight up a tiny bit, does the cost go up or down, and by how much?"** That answer is the **gradient**, written $\frac{\partial C}{\partial w}$.

- Gradient positive → raising the weight raises the cost → **lower** the weight.
- Gradient negative → raising the weight lowers the cost → **raise** the weight.

So every weight and bias moves against its gradient:

$$
w_{\text{new}} = w_{\text{old}} - \eta \, \frac{\partial C}{\partial w}
\qquad\qquad
b_{\text{new}} = b_{\text{old}} - \eta \, \frac{\partial C}{\partial b}
$$

$\eta$ (eta) is the **learning rate**, the size of each step ([Section 10](#10-the-learning-rate)).

This project uses **stochastic gradient descent (SGD)**: the weights are updated after every single image.

### The chain rule

A weight never affects the cost directly. Its effect travels along a chain:

```
weight  -->  z  -->  output (a)  -->  cost
```

The **chain rule** says the total effect is each link's effect multiplied together

| Link | Derivative | Meaning |
| --- | --- | --- |
| weight → z | the input that weight multiplies | A large input means that weight had a large effect |
| z → a | $a(1 - a)$ | The sigmoid's slope: how responsive the neuron is |
| a → cost | $2(a - y)$ | How wrong the output was, and in which direction |

The sigmoid's slope is $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. Since $\sigma(z)$ is just the neuron's output $a$, this becomes $a(1 - a)$, which can be calculated straight from the output. Its maximum is 0.25.

So for a weight going into an output neuron:

$$
\frac{\partial C}{\partial w} = \text{input} \cdot a(1 - a) \cdot 2(a - y)
$$

### Delta (δ): the shortcut

Every weight going into the same neuron shares the last two parts of that formula; only the input differs. So the shared part is calculated **once per neuron** and called **delta**:

> **δ = how much the cost changes when this neuron's $z$ changes**

Once a neuron's delta is known, its gradients are simple:

$$
\frac{\partial C}{\partial w} = \delta \times \text{input}
\qquad\qquad
\frac{\partial C}{\partial b} = \delta
$$

The bias gradient is just δ, because nudging the bias changes $z$ by exactly the same amount.

**Backpropagation** is the algorithm that calculates every neuron's delta, then uses them to update every weight.

## 9. Backpropagation

### Neurons need to remember

Every formula above uses a neuron's **output** $a$ and its **inputs**. These only exist during the forward pass, so each neuron now stores them, along with a place for its delta:

```python
class Neuron:
    def __init__(self, num_inputs):
        ...
        # Each neuron will store its input and output, so it can be seen for backpropagation
        self.input = None
        self.output = None
        self.delta = 0

    def feedforward(self, inputs):
        total = np.dot(inputs, self.weight) + self.bias
        out = sigmoid_function(total)

        self.input = inputs
        self.output = out

        return out
```

Backpropagation for one image then runs in four steps, all inside one method:

```python
def backpropagation(self, inputs, hot_encoded, learning_rate):
```

### Step 1: Forward pass

```python
# Runs feedforward at start, so every neuron stores its input and output
self.feedforward(inputs)
```

### Step 2: Output-layer deltas

```python
for j, neuron in enumerate(self.layers[-1]):
    a = neuron.output
    y = hot_encoded[j]

    neuron.delta = a*(1-a) * 2*(a-y)
```

$$
\delta_j = 2(a_j - y_j) \cdot a_j(1 - a_j)
$$

- $2(a - y)$: how wrong the output was. Positive means too high, negative means too low.
- $a(1 - a)$: how responsive the neuron is. A saturated neuron gets a tiny delta.
- `enumerate` gives each output neuron's position `j`, which picks its target from the one-hot label.

**Why the output layer comes first:** it's the only layer with targets, so its deltas are the only ones that can be calculated directly from the label.

**For the first image (a 5):**

```
[0.2627, 0.2073, 0.2942, 0.2888, 0.2800, -0.1492, 0.1696, 0.1892, 0.2959, 0.2925]
```

Only neuron 5's delta is negative (its output is too low, so it should rise). The other nine are positive (theirs should fall).

### Step 3: Hidden-layer deltas, going backwards

```python
# Start at the last hidden layer and move backwards to the 1st hidden layer
for i in range(len(self.layers) - 2, -1, -1):
    current_layer = self.layers[i]  # Hidden layer whose deltas are being calculated
    next_layer = self.layers[i+1]   # The layer after, whose deltas are already calculated

    for j, neuron in enumerate(current_layer):
        blame = 0   # Accumulates the sum: sum(weight * delta)

        for next_neuron in next_layer:
            # The weight connecting the current neuron (index j) to the next neuron
            blame += next_neuron.weight[j] * next_neuron.delta

        a = neuron.output
        sigmoid_derivative = a*(1-a)

        neuron.delta = sigmoid_derivative * blame
```

$$
\delta_k = \Big( \sum_j w_{jk} \, \delta_j \Big) \cdot a_k(1 - a_k)
$$

**The idea:** a hidden neuron has no target, so it can't use $(a - y)$. It is only "wrong" because of the effect it had on the neurons it feeds into. Its output reaches **every** neuron in the next layer, through a different weight each time:

- Each path contributes **weight × that neuron's delta**: how much influence it had × how wrong that neuron was.
- All the paths are **added up**, because they all affect the cost at the same time. That total is the `blame`.
- It's multiplied by the neuron's own sigmoid slope, to go back one more link, from $a$ to $z$.

**It mirrors the forward pass:** going forwards, a neuron adds up (weight × input) from the previous layer. Going backwards, it adds up (weight × delta) from the next layer.

**Why backwards:** each layer's deltas need the **next** layer's deltas to already exist. Hidden layer 2 uses the output layer's deltas, and hidden layer 1 then uses hidden layer 2's. The error **propagates backwards** through the network, which gives the algorithm its name. This is why the layers are stored in one list ([Section 5](#building-the-layers)).

**The loop:** `range(start, stop, step)` with a step of −1 counts down. `self.layers` has 3 items (indices 0, 1, 2), and index 2 is the output layer, already done. So `range(len(self.layers) - 2, -1, -1)` starts at **1**, counts down, and stops *before* −1, giving **1, then 0**. It's built from `len(self.layers)`, so it still works if more layers are added.

**Why `next_neuron.weight[j]`:** each neuron stores its **incoming** weights, one per neuron in the previous layer, in order. So the weight connecting hidden neuron `j` to a next-layer neuron is stored inside **that next-layer neuron**, at index `j`.

**For the first image:**

| Layer | First 3 deltas |
| --- | --- |
| Hidden 2 (`self.layers[1]`) | `[0.0413, -0.0352, 0.1837]` |
| Hidden 1 (`self.layers[0]`) | `[0.0008, -0.0172, -0.0078]` |

**Vanishing gradients:** the deltas shrink as they travel backwards, because each layer multiplies by a sigmoid slope of at most 0.25. So earlier layers get weaker signals and learn more slowly. With 10 layers, $0.25^{10} \approx 0.000001$, and the first layers would barely learn. This is why deep networks often use ReLU instead of sigmoid in hidden layers. With two hidden layers, sigmoid works well here.

### Step 4: Update every weight and bias

```python
for layer in self.layers:
    for neuron in layer:
        # Calculate gradient
        gradient = np.array(neuron.input) * neuron.delta

        neuron.weight -= learning_rate * gradient
        neuron.bias -= learning_rate * neuron.delta
```

- `input × delta` calculates the gradient for **all** of a neuron's weights at once.
- `-=` moves **downhill**, against the gradient.
- The bias's gradient is just its delta.
- **`np.array(...)`:** the forward pass passes each layer's outputs on as a Python list, so most neurons store their inputs as lists. A list times a number doesn't do maths in Python (`[1, 2] * 3` gives `[1, 2, 1, 2, 1, 2]`, and a list times a decimal crashes). Converting to a NumPy array multiplies each element instead.

**Why update last:** Step 3 uses the next layer's weights to share out the blame. Those must be the weights that actually made this prediction. If the output weights were updated first, the hidden layers would be blamed using weights that weren't there when the prediction was made, giving the wrong gradients. So **all deltas are calculated first, then all weights are updated**.

### Does it work?

Training on the first image alone, 20 times in a row, its cost falls steadily:

```
2.69 → 2.37 → 2.09 → 1.84 → 1.62 → 1.43 → 1.26 → 1.12 → 0.99 → 0.89
     → 0.80 → 0.72 → 0.65 → 0.59 → 0.54 → 0.49 → 0.45 → 0.42 → 0.39 → 0.36
```

A steady fall shows the gradients really do point downhill: **the network is learning.** How fast it learns depends on the size of each step:

## 10. The learning rate

```python
# Hyperparameter. Used to calculate the new weight.
LEARNING_RATE = 0.1
```

The learning rate controls how big each step downhill is. It's a **hyperparameter**: a setting chosen by the programmer, not learned by the network.

| Too small | Good | Too big |
| --- | --- | --- |
| Always goes downhill, but takes forever | Steady, fast progress | Overshoots the valley and bounces around, or never learns |

**How 0.1 was chosen:**
By experiment. Training this network for 3 epochs on 10,000 images with different learning rates gave these accuracies on images it hadn't trained on:

| Learning rate | Epoch 1 | Epoch 2 | Epoch 3 |
| --- | --- | --- | --- |
| 0.001 | 11.5% | 10.2% | 10.2% |
| 0.01 | 11.1% | 35.1% | 53.2% |
| **0.1** | **84.5%** | **89.1%** | **90.7%** |
| 0.5 | 86.0% | 85.4% | 88.2% |
| 2 | 65.8% | 69.3% | 65.5% |
| 10 | 9.5% | 10.0% | 9.5% |

The best value depends on the network, the activation function and the cost function, so it's always found by testing, usually trying powers of 10 first.

## 11. Running the code

**Requirements:** Python 3, NumPy, pandas.

```bash
pip install numpy pandas
```

1. Download `mnist_train.csv` (and `mnist_test.csv`) into the project folder.
2. Run:

```bash
python main.py
```

**Expected output before learning** (with `np.random.seed(0)`):

| Check | Value |
| --- | --- |
| Outputs for the first image | `[0.527, 0.424, 0.634, 0.604, 0.753, 0.665, 0.366, 0.396, 0.681, 0.622]` |
| Cost for the first image | ≈ 3.04 |
| Accuracy (first 1,000 images) | 10.5% |
| Average cost (first 1,000 images) | ≈ 3.25 |