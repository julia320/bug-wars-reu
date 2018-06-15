# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 16:20:45 2018

@author: Julia
"""

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering

# initialize random # generator to a constant
# this ensures reproducability
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

#X_train = X_train.reshape(X_train.shape[0], 1,28,28).astype('float32')
#X_test = X_test.reshape(X_test.shape[0], 1,28,28).astype('float32')

# normalize inputs from 0-225 to 0-1
X_train = X_train/255
X_test = X_test/255

# turn the vector of possible outputs into a binary matrix
# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# now we can make the neural network
# define the baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal'))
    model.add(Activation('relu'))
    model.add(Dense(num_classes, kernel_initializer='normal'))
    model.add(Activation('softmax'))
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("fuck again")
    return model

# build the model
model = baseline_model()
# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200, verbose=2)
# final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Error: %.2f%%" % (100-scores[1]*100))