# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:04:53 2018

@author: Julia
"""

import numpy
from keras.models import load_model
from keras.datasets import cifar10
from keras.utils import np_utils

model = load_model('cifar10_deep_cnn_relu.h5')

seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# normalize inputs from 0-225 to 0-1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255
X_test = X_test/255

# turn the vector of possible outputs into a binary matrix - one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#model.predict(model, X_test, verbose=1)

scores = model.evaluate(X_test, y_test, verbose=1)
print(scores) 
print("Error: %.2f%%" % (100-scores[1]*100))