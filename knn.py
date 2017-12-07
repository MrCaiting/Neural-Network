from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import time

# global variables
# give a range to K value
k = [5, 11]

# read train and test data
train = pd.read_csv(r'cnn_data/cnn_train.txt', header=None, skipinitialspace=True, skip_blank_lines=True, sep=',').values
train = train.astype(np.float)

train_label = pd.read_csv('cnn_data/cnn_trainlabel.txt', header=None).values.ravel()

test = pd.read_csv(r'cnn_data/cnn_test.txt', header=None, skipinitialspace=True, skip_blank_lines=True, sep=',').values
test = test.astype(np.float)

test_label = pd.read_csv('cnn_data/cnn_testlabel.txt', header=None).values.ravel()


def KNN(k):
    # declare classifier
    classifier = KNeighborsClassifier(n_neighbors=k)
    start = time.clock()
    # fit the model
    classifier.fit(train, train_label)
    train_time = time.clock() - start
    # predict label
    predict_label = classifier.predict(test)
    # compare labels
    accuracy = np.mean(np.equal(test_label, predict_label)) * 100

    return accuracy, train_time


for k_val in range(k[0], k[1]+1):
    accuracy, t_time = KNN(k_val)
    print("K Nearest Neighbors with K =", k_val)
    print("Training Time: ", t_time, "seconds")
    print("Accuracy: ", accuracy, "%")