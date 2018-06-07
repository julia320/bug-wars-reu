# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:04:53 2018

@author: Julia
"""

import numpy
from keras.models import load_model
from keras.datasets import mnist

model = load_model('mnist_cnn.h5')

seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape the images to 28x28
X_train = X_train.reshape(X_train.shape[0], 28,28,1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28,28,1).astype('float32')
# normalize inputs from 0-225 to 0-1
X_train = X_train/255
X_test = X_test/255

model.predict(model, X_test, verbose=1)