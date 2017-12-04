import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd


# define all functions for use
# set up one-hot vectors
def one_hot(labels, numbers):
   num_of_labels = len(labels)  # how many labels in total
   one_hot = np.zeros((num_of_labels, numbers))  # set up a matrix with 42000 rows and 10 columns
   adjustment = np.arange(num_of_labels) * numbers  # set up a vector that looks like [0, 10, 20,..., 42000]
   one_hot.flat[adjustment + labels] = 1
   return one_hot

# weight initialization
def weight_generate(shape):
   initial = tf.truncated_normal(shape, stddev=0.1)
   return tf.Variable(initial)

def bias_generate(shape):
   initial = tf.constant(0.1, shape=[shape])
   return tf.Variable(initial)

# function to set up a new convolution layer and do max-pooling
def convolution_layer(input, num_channel, size_filter, num_filter):
   # define the shape of the weight function
   shape = [size_filter, size_filter, num_channel, num_filter]
   print(input.shape)
   # generate weights
   weights = weight_generate(shape)
   # generate bias
   bias = bias_generate(shape=num_filter)
   # conduct the convolution operation of the weights and the data
   convolution = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
   # add the bias to the result of the convolution
   convolution = convolution + bias
   # max_pooling
   convolution = tf.nn.max_pool(convolution, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
   # Rectified linear unit: max(x,0)
   convolution = tf.nn.relu(convolution)
   return convolution

def flatten_layer(input):
   # get the shape of the input
   shape = input.get_shape()
   # number of features is image height times image width times number of input channels
   num_feature = shape[1:4].num_elements()
   print(num_feature)
   # reshape the input to flatten it
   flatten = tf.reshape(input, [-1, num_feature])
   return flatten, num_feature

def fully_connected_layer(flat_conv_output, num_features, num_neurons):
   # generate new weight and bias for fc layer
   weight_fc = weight_generate([num_features, num_neurons])
   bias_fc = bias_generate(shape=num_neurons)
   # matrix multiplication of flattened output and new weight, add bias
   # use ReLU here to allow non linearity
   fc_layer = tf.nn.relu(tf.matmul(flat_conv_output, weight_fc) + bias_fc)
   return fc_layer

def softmax_layer(num_neurons, classes, fc_layer):
   # generate new weight and bias for softmax layer
   weight_sm = weight_generate([num_neurons, classes])
   bias_sm = bias_generate(shape=classes)

   # softmax layer to get a shape (num_input,num_labels)
   # no ReLU
   sm_layer = tf.nn.softmax(tf.matmul(fc_layer, weight_sm) + bias_sm)
   return sm_layer



# import data
# define data dimensions
image_size = 28
HEIGHT = 28
WIDTH = 28
TOTAL_PIXEL = HEIGHT * WIDTH
image_shape = (image_size, image_size)
image_flattened = image_size * image_size

# separate label and training set
label_temp = open('digitdata/traininglabels', 'r')
train_label_for_cnn = open('cnn_data/cnn_trainlabel.txt', 'w')

for char in label_temp:
    if char != '\n':
        train_label_for_cnn.write('%s'%char)
label_temp.close()
train_label_for_cnn.close()

label = pd.read_csv('cnn_data/cnn_trainlabel.txt', header=None).values.ravel()


train_temp = open('digitdata/trainingimages', 'r')
train_for_cnn = open('cnn_data/cnn_train.txt', 'w')
text_counter = 0

for line in train_temp.readlines():
    for char in line:
        if char != '\n':
            text_counter += 1
            if char == '+' or char == '#':
                train_for_cnn.write('1')
                if text_counter == TOTAL_PIXEL:
                    train_for_cnn.write('\n')
                    text_counter = 0
                elif char != '\n':
                    train_for_cnn.write(',')
            elif char == ' ':
                train_for_cnn.write('0')
                if text_counter == TOTAL_PIXEL:
                    train_for_cnn.write('\n')
                    text_counter = 0
                elif char != '\n':
                    train_for_cnn.write(',')
train_temp.close()
train_for_cnn.close()

train = pd.read_csv(r'cnn_data/cnn_train.txt', header=None, skipinitialspace=True, skip_blank_lines=True, sep=',').values

train = train.astype(np.float)


print("Training Data Labels: ", label)
print("Number of Labels: ", len(label))

# process the test data
test_temp = open('digitdata/testimages', 'r')
test_for_cnn = open('cnn_data/cnn_test.txt', 'w')
text_counter = 0
for line in test_temp.readlines():
    for char in line:
        if char != '\n':
            text_counter += 1
            if char == '+' or char == '#':
                test_for_cnn.write('1')
                if text_counter == TOTAL_PIXEL:
                    test_for_cnn.write('\n')
                    text_counter = 0
                elif char != '\n':
                    test_for_cnn.write(',')
            elif char == ' ':
                test_for_cnn.write('0')
                if text_counter == TOTAL_PIXEL:
                    test_for_cnn.write('\n')
                    text_counter = 0
                elif char != '\n':
                    test_for_cnn.write(',')
test_temp.close()
test_for_cnn.close()


test = pd.read_csv(r'cnn_data/cnn_test.txt', header=None, skipinitialspace=True, skip_blank_lines=True, sep=',').values
test = test.astype(np.float)

print("Training Data Size: ", train.shape)

print("Testing Data Size: ", test.shape)
# find out how many different classes there are
classes = len(np.unique(label))
print("Number of Unique Classes: ", classes)

# convert labels to a one_hot vector
one_hot_labels = one_hot(label, classes)

# make placeholders for the output of the first layer to go to the second layer
placeholder1 = tf.placeholder('float', shape=[None, image_flattened])
y = tf.placeholder('float', shape=[None, classes])

# reshape the placeholder
x = tf.reshape(placeholder1, [-1, image_size, image_size, 1])

# define convolution parameters
num_of_channels = 1
size_filter = 5
num_filter1 = 32
num_filter2 = 64
number_neurons = 1024
learning_rate = 1e-4

# first convolutional layer with pooling
first_conv = convolution_layer(x, num_of_channels, size_filter, num_filter1)

# second convolutional layer with pooling
second_conv = convolution_layer(first_conv, num_filter1, size_filter, num_filter2)

# flatten the output layer
flat_layer, num_of_features = flatten_layer(second_conv)

# fully connected layer
fully_connected = fully_connected_layer(flat_layer, num_of_features, number_neurons)

# softmax layer
softmax_result = softmax_layer(number_neurons, classes, fully_connected)

# cost function (for reducing the training loss)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=softmax_result, labels=y)

cost = tf.reduce_sum(cross_entropy)

# optimization
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# prediction function
prediction = tf.argmax(softmax_result, 1)

# evaluation
y_correct_predictions = tf.equal(prediction, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(y_correct_predictions, 'float'))



# training process
# iterations
iterations = 5000

# batch size
batch_size = 128

# progress in dealing with the batches
entirely_completed = 0  # number of times that we have processed the entire dataset
batch_index = 0  # the index indicating how many pictures in total have been processed
len_train = train.shape[0]

# serve by batches
def feed_data(size):
   # take in the global variables
   global entirely_completed
   global batch_index
   global train
   global one_hot_labels

   # keep track of where we are in terms of batches
   start = batch_index
   batch_index += size

   # if we trained so many batches that we reach the end of the dataset we randomly shuffle the dataset and restart
   if batch_index > len_train:
       # we have finished the entire data set one more time
       entirely_completed += 1
       # generate integer from 0 to 41999
       permutation = np.arange(len_train)
       # shuffle the dataset randomly
       np.random.shuffle(permutation)
       train = train[permutation]
       one_hot_labels = one_hot_labels[permutation]
       # start processing the newly shuffled dataset
       start = 0
       batch_index = size
   end = batch_index
   processed_data = np.reshape(train[start:end], [-1, image_size, image_size, 1])
   return processed_data, one_hot_labels[start:end]

# start the TensorFlow session (Kaggle version is outdated)
session = tf.Session()
session.run(tf.global_variables_initializer())
train_accuracy = []
step = 100
for i in range(iterations):
 # get data
 train_data, label_data = feed_data(batch_size)
 # training
 #jump out condition
 if i%step == 0 or (i+1) == iterations:
     train_accuracy = accuracy.eval(session=session, feed_dict={x:train_data, y:label_data})
     print(train_accuracy)
 session.run(optimizer, feed_dict={x: train_data, y: label_data})

# reshape test data ready for placeholder
train_1 = np.reshape(train, [-1,image_size, image_size, 1])
test = np.reshape(test,[-1, image_size, image_size, 1])
# create a numpy array to store the predicted labels
prediction_results = prediction.eval(feed_dict={x: train_1}, session=session)

print('prediction_results({0})'.format(len(prediction_results)))

def display(img):
  # (784) => (28,28)
  one_image = img.reshape(image_size, image_size)

  plt.axis('off')
  plt.imshow(one_image, cmap='binary')
  plt.show()







# create a numpy array to store the predicted labels
prediction_results = prediction.eval(feed_dict={x: test}, session=session)


# read test labels
test_label = open('digitdata/testlabels', 'r')
test_label_for_cnn = open('cnn_data/cnn_testlabel.txt', 'w')

for char in test_label:
    if char != '\n':
        test_label_for_cnn.write('%s'%char)
test_label.close()
test_label_for_cnn.close()

testing_label = pd.read_csv('cnn_data/cnn_testlabel.txt', header=None).values.ravel()

#calculate accuracy
n_correct = 0
for index in range(len(testing_label)):
    if testing_label[index] == prediction_results[index]:
        n_correct += 1

final_accuracy = n_correct * 100/len(testing_label)

print('Accuracy: ', final_accuracy, '%')