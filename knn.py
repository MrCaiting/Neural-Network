from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import time

# global variables
# give a range to K value
k = [1, 11]
MODES = ['ball_tree', 'kd_tree','brute','auto']
mode = MODES[2]
TOTAL_DIG = 10
print("Current Search Algorithm:", mode)

# read train and test data
train = pd.read_csv(r'cnn_data/cnn_train.txt', header=None, skipinitialspace=True, skip_blank_lines=True, sep=',').values
train = train.astype(np.float)

train_label = pd.read_csv('cnn_data/cnn_trainlabel.txt', header=None).values.ravel()

test = pd.read_csv(r'cnn_data/cnn_test.txt', header=None, skipinitialspace=True, skip_blank_lines=True, sep=',').values
test = test.astype(np.float)

test_label = pd.read_csv('cnn_data/cnn_testlabel.txt', header=None).values.ravel()


def KNN(k):
    # declare classifier
    classifier = KNeighborsClassifier(n_neighbors=k, algorithm=mode)
    start = time.clock()
    # fit the model
    classifier.fit(train, train_label)
    train_time = time.clock() - start
    # predict label
    predict_label = classifier.predict(test)
    # compare labels
    accuracy = np.mean(np.equal(test_label, predict_label)) * 100

    return accuracy, train_time, predict_label

temp_predict_label = []
temp_max = None
best_k = None

for k_val in range(k[0], k[1]+1):
    accuracy, t_time, predict_label = KNN(k_val)
    print("K Nearest Neighbors with K =", k_val)
    print("Training Time: ", t_time, "seconds")
    print("Accuracy: ", accuracy, "%")
    if temp_max == None or accuracy > temp_max:
        temp_max = accuracy
        best_k = k_val
        temp_predict_label = predict_label
    else:
        continue



def confusion_matrix(y, y_, length):
    conf_m = np.zeros((TOTAL_DIG, TOTAL_DIG))
    for number in range(TOTAL_DIG):
        number_counter = 0
        for index in range(length):
            if y[index] == number:
                number_counter += 1
            if y[index] == number and y_[index] == number:
                conf_m[number][number] += 1
            elif y[index] == number and y_[index] != number:
                conf_m[number][y_[index]] += 1
        conf_m[number] = conf_m[number] * 100 / number_counter

    return conf_m
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
print("\n\nFound Best Performance with K = ", best_k)
print("The Confusion Matrix with K = ", best_k, "is:\n")
conf_m = confusion_matrix(test_label, temp_predict_label, len(test_label))
print(conf_m)