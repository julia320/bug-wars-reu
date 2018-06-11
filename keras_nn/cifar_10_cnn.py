# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:45:03 2018

@author: Julia
"""

import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering

# initialize random # generator to a constant
# this ensures reproducability
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
num_classes = y_test.shape[1]


# now we can make the neural network
# define the baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32,(3,3), input_shape=(32,32,3), padding='same'))
    model.add(Activation('relu'))
    # Dropout helps prevent overfitting
    model.add(Dropout(0.2))
    
    model.add(Conv2D(32,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(128,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(128,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    
    # squishes the data
    model.add(Flatten())
    model.add(Dropout(0.2))
    
    # Fully connected layers
    model.add(Dense(1024, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.add(Dropout(0.2))
    
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = baseline_model()
# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=200, verbose=1)
# save the model
model.save('cifar10_deep_cnn_relu.h5')
# final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Error: %.2f%%" % (100-scores[1]*100))