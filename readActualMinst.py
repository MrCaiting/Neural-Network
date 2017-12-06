import cPickle
import gzip

# Third-party libraries
import numpy as np


def readMnist():
    f = gzip.open('digitdata/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)
