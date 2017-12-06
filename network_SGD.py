"""network_SGD."""
import random
import numpy as np
from readActualMinst import readMnist

"""
This file contains method implemented by using Sigmoid perceptrons and
    (stochastic) gradient descent rule for the logistic regression to build a
    feedforward network with hidden layers.
"""
TRAIN_LABEL = 'digitdata/traininglabels'
TRAIN_DATA = 'digitdata/trainingimages'
TEST_LABEL = 'digitdata/testlabels'
TEST_DATA = 'digitdata/testimages'
WIDTH = 28
HEIGHT = 28
TOTAL_PIXEL = WIDTH*HEIGHT
TOTAL_DIG = 10
TOTAL_IMG = 5000

# Chaning Constant for getting best result
HIDDEN = 100
EPOCHS = 30
SIZE = 10
LEARN_RATE = 3.0


# Here we need to establish some helper functions for later use
def sigmoid(vector):
    """ Function that is used to calculated sigmoid function value"""
    return 1. / (1. + np.exp(-vector))


def deriv_sig(vector):
    """deriv_sig.

    DESCRIPTION:
        Function used to calculated first derivative of sigmoid function.
        Based on the derivative property of sigmoid function from:
        https://beckernick.github.io/sigmoid-derivative-neural-network/
    """
    return sigmoid(vector) * (1 - sigmoid(vector))


# The main structure of our nural network
class NeuralNetwork(object):

    def __init__(self, struct_param):
        """__init__.

        DESCRIPTION:
            Function used to initialize a class of nural network.
        INPUTS:
            struct_param: a list consisting the numbers of neurons on different layers
        """
        self.layers = 3     # Set to constant since layers are at most three
        self.structure = struct_param
        # Start filling the above two lists, and both used standard normal random function
        self.weights = [np.random.randn(y, x) for x, y in zip(struct_param[:-1], struct_param[1:])]
        self.biases = [np.random.randn(x, 1) for x in struct_param[1:]]

        """
        for y, x in zip(struct_param[1:], struct_param[:-1]):
            self.weights.append(np.random.randn(y, x))
        for x in struct_param[1:]:      # Strip the first one since input has no bias
            self.biases.append(np.random.randn(x, 1))
        """

    def feedForward(self, action):
        for bias, weight in zip(self.biases, self.weights):
            action = sigmoid(np.dot(weight, action) + bias)
        return action

    def update_SDG(self, batch, learning_rate):
        """update_SDG.

        DESCRIPTION:
            This function is used to update all the weights and biases every time
                we are looping inside all the epochs
        """
        batch_size = len(batch)
        # bath is a list of tuple here

        #print("Length: ", batch)
        # These are lists of biases and weights used for getting from the current
        #   batch and use them to update the overall trained biases and weights
        grad_biases = []
        grad_weights = []

        # Initialize these list to correct size filled with zeros
        for bias in self.biases:
            grad_biases.append(np.zeros(bias.shape))
        for weight in self.weights:
            grad_weights.append(np.zeros(weight.shape))

        for image, label in batch:
            delta_b, delta_w = self.back_propagate(image, label)

            # Start updating the gradient of bias and weights
            grad_biases = [gb + db for gb, db in zip(grad_biases, delta_b)]
            grad_weights = [gw + dw for gw, dw in zip(grad_weights, delta_w)]

            # Change the biases and weights in the NeuralNetwork class
            self.biases = [b - (learning_rate/batch_size)*gb for
                           b, gb in zip(self.biases, grad_biases)]
            self.weights = [w - (learning_rate/batch_size)*gw for
                            w, gw in zip(self.weights, grad_weights)]

    def back_propagate(self, image, label):
        """back_propagate.

        """
        #print("Wonder How Many Times")
        grad_biases = [np.zeros(bias.shape) for bias in self.biases]
        grad_weights = [np.zeros(weight.shape) for weight in self.weights]
        activation = image
        activations = [image]
        act_vectors = []
        for bias, weight in zip(self.biases, self.weights):
            #print("HERE: ", weight.shape)
            vector = np.dot(weight, activation) + bias
            act_vectors.append(vector)
            activation = sigmoid(vector)
            activations.append(activation)

        # Calculate the change of delta C based on the last element in each list
        #   This is why the algorithm is callled back propagation
        delta = (activations[-1] - label) * deriv_sig(act_vectors[-1])

        grad_biases[-1] = delta
        grad_weights[-1] = np.dot(delta, activations[-2].transpose())

        for layer in xrange(2, self.layers):
            #print("I dont think I can be here twice seriously, l = ", layer)
            vector = act_vectors[-layer]
            d_sig = deriv_sig(vector)
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * d_sig
            grad_biases[-layer] = delta
            grad_weights[-layer] = np.dot(delta, activations[-layer-1].transpose())

        return grad_biases, grad_weights

    def train_SDG(self, training_data, epochs, batch_size, learning_rate, test_data=None):
        if test_data:
            num_test = len(test_data)
        num_train = len(training_data)
        # print("Length: ", training_data[0])

        # iterate the times specified by epoch value
        for times in range(epochs):
            # Randomize the list of training data
            random.shuffle(training_data)

            # Since we are using stochastic gradient descent, we need to
            #   randomly choose some of the data
            batches = [training_data[i:i+batch_size]
                       for i in xrange(0, num_train, batch_size)]

            for batch in batches:
                self.update_SDG(batch, learning_rate)
            if test_data:
                print("Epoch time: ", times, ", Correct over Total: ",
                      self.validate(test_data), " / ", num_test)
            else:
                print("Training has finished, total epochs: ", times)

    def validate(self, test_data):
        results = [(np.argmax(self.feedForward(image)), label) for image, label in test_data]
        """
        for image, label in test_data:
            results.append((np.argmax(self.feedForward(image)), label))
        """
        correct_count = 0
        for ans, correct in results:
            #print("what are they? ", ans, correct)
            if ans == correct:
                correct_count += 1
            else:
                correct_count += 0
        return correct_count


def vectorized_result(digit):
    v_r = np.zeros((10, 1))
    v_r[digit] = 1.0
    return v_r

"""
# Read training labels
with open(TRAIN_LABEL, 'r') as train_l:
    t_labels = [int(x.strip('\n')) for x in train_l.readlines()]

with open(TRAIN_DATA, 'r') as train_d:
    image = [y.strip('\n') for y in train_d.readlines()]

# Read training data
training_data = []
for index in range(TOTAL_IMG):
    # starting reading every single picture
    this_image = []
    # parse every single digit image into decimal value instead of char
    for i in range(index*HEIGHT, (index+1)*HEIGHT):
        for pixel in image[i]:
            if pixel == "#":
                this_image.append(1.0)
            # if it is a plus sign, which denote the border, we assign 0.5
            elif pixel == "+":
                this_image.append(1.0)
            # otherwise, we do move on without doing anything
            else:
                this_image.append(0.0)
    training_data.append((np.reshape(this_image, (TOTAL_PIXEL, 1)),
                          vectorized_result(t_labels[index])))
"""

# Read test labels (Need this for testing)
testlabels = open(TEST_LABEL, 'r')
test_table = [int(x.strip('\n')) for x in testlabels.readlines()]

# Read test data
testd = open(TEST_DATA, 'r')
test_data = []
curr_list = []
index = 1
for line in testd.readlines():
    for char in line:
        if char == ' ':
            curr_list.append(0.0)
        elif char == '+':
            curr_list.append(1.0)
        elif char == '#':
            curr_list.append(1.0)

    if len(curr_list) == TOTAL_PIXEL:
        test_data.append((np.reshape(curr_list, (TOTAL_PIXEL, 1)),
                          test_table[(index/HEIGHT) - 1]))
        curr_list = []
    index += 1

# An alternative way to import MNIST data
t_d, _, test = readMnist()
train_in = [np.reshape(pic, (784, 1)) for pic in t_d[0]]
train_out = [vectorized_result(x) for x in t_d[1]]
training_data_new = zip(train_in, train_out)

# Not quite necessary for testing
test_in = [np.reshape(sample, (784, 1)) for sample in test[0]]
test_data_new = zip(test_in, test[1])


"""
# Used for tranditional method
network = NeuralNetwork([TOTAL_PIXEL, HIDDEN, TOTAL_DIG])
network.train_SDG(training_data, EPOCHS, SIZE, LEARN_RATE, test_data)
"""

network = NeuralNetwork([TOTAL_PIXEL, HIDDEN, TOTAL_DIG])
network.train_SDG(training_data_new, EPOCHS, SIZE, LEARN_RATE, test_data)
