# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:39:48 2018

@author: Julia

@purpose: generate text files with slightly varying ML architectures 
"""


import os
from keras.datasets import mnist
from keras.utils import np_utils
# import activation functions
import polynomials


# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# ask for the file path they want
message = "Where do you want these files stored? (Enter the full path, no slashes at the end): "
path = input(message)
wrong_count = 0
while not os.path.exists(path):
    wrong_count+=1
    path = input("That path does not exist on your computer, please try again: ")
	if wrong_count > 2:
        print("Here is an example of an acceptable path: C:\\Users\\Julia\\REU")

# make the list of polynomials
activations = polynomials.polynomials
for act in activations:
    # make a folder for each polynomial
    file_path = path + "\\" + str(act)
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    
    # navigate to the folder just created
    os.chdir(file_path)
    
    # make text files 
    # each file needs a FC layer with i nodes and an output layer
    for i in range(100):
        # create the file
        file_name = str(act) + "_" + str(i+1) + ".txt"
        file = open(file_name, "w+")
        
        # model.add(Dense(i, input_dim=num_pixels, kernel_initializer='normal', activation='polynomial'))
        file.write("Dense\t" + str(i+1) + "\tidm=" + str(num_pixels) + "\tini=normal")
        file.write("\nActiv\t" + str(act))
        # model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax')) 
        file.write("\nDense\t" + str(num_classes) + "\tini=normal")
        file.write("\nActiv\tsoftmax")
        
        print("File " + file_name + " was created!")
        
        # close the file
        file.close()
    
    # reset the directory path
    os.chdir(path)
    
print("All folders and text files created.")