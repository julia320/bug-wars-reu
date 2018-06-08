# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 16:09:26 2018

@author: Julia
"""

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
K.set_image_dim_ordering


# initialize random # generator to a constant
# this ensures reproducability
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape the images to 28x28
X_train = X_train.reshape(X_train.shape[0], 28,28,1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28,28,1).astype('float32')
y_train = y_train.reshape(y_train.shape[0], 28,28,1).astype('float32')
y_test = y_test.reshape(y_test.shape[0], 28,28,1).astype('float32')

# normalize inputs from 0-225 to 0-1
X_train = X_train/255
X_test = X_test/255

# turn the vector of possible outputs into a binary matrix
# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# activation function for squaring a number
def square_activation(x):
    return x**2

# add square activation function to keras
get_custom_objects().update({'square_activation': Activation(square_activation)})


# now we can make the neural network
# define the baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(32, input_shape=(28,28,1)))
    model.add(Activation(square_activation))
    print("using the square activation!")
    
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = baseline_model()

# fit the model
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200, verbose=1)

# save the model
model.save('square_activation.h5')

# final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Error: %.2f%%" % (100-scores[1]*100))