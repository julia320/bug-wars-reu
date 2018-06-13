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
# find the number of output classes
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# make the list of polynomials
activations = polynomials.polynomials
for act in activations:
    # make a folder for each polynomial
    path = "C:\\Users\\Julia\\Code\\Bug Wars\\bug-wars-reu\\test-poly-accuracy\\" + str(act)
    directory = os.path.dirname(path)
    if os.path.exists(path) == False:
        os.mkdir(path)
    
    # navigate to the folder just created
    os.chdir(path)
    
    # make text files 
    # each file needs an input and output layer and a FC layer with i nodes
    for i in range(100):
        # create the file
        file_name = str(act) + "_" + str(i) + ".txt"
        file = open(file_name, "w+")
        
        # write the input Conv2D layer to the file
        file.write("Conv2D\t32\t5,5")
        # write the activation function
        file.write("\nActiv\t" + str(act))
        # write the fully connected layer with i nodes
        file.write("\nDense\t" + str(i))
        # write another activation layer 
        file.write("\nActiv\t" + str(act))
        # write the output layer 
        file.write("\nDense\t10")
        # write the final activation layer
        file.write("\nActiv\tsoftmax")
        
        print(file_name + " was created!")
        
        # close the file
        file.close()
    
    # reset the directory path
    os.chdir("C:\\Users\\Julia\\Code\\Bug Wars\\bug-wars-reu\\test-poly-accuracy")
    
print("All folders and text files created.")