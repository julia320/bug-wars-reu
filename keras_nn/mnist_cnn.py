# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:26:03 2018

@author: Julia Bristow
"""

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering

# initialize random # generator to a constant
# this ensures reproducability
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape the images to 28x28
X_train = X_train.reshape(X_train.shape[0], 1,28,28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1,28,28).astype('float32')

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
    model.add(Conv2D(32, (5,5), input_shape=(1,28,28)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    # Dropout helps prevent overfitting
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = baseline_model()
# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200, verbose=2)
# save the model
model.save('mnist_cnn.h5')
# final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Error: %.2f%%" % (100-scores[1]*100))