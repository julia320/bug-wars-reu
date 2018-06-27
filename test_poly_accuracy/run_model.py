# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 14:05:35 2018

@author: Julia

@purpose: create and run an ML model from a text file with the architecture
"""

import os
# imports for making a model from a text file
import cMod
import load

file_path = input("Enter the full path of your text file: ")
wrong_count = 0
while not os.path.exists(file_path):
    wrong_count+=1
    if wrong_count > 2:
        print("Here is an example of an acceptable path: C:\\Users\\Julia\\Bug Wars\\example.txt")
    file_path = input("That file does not exist, please try again: ")

model_name = input("Enter a name for the saved model (no extension needed): ") + ".h5"
    

def run_model ():
    #load mnist dataset
    X_train, y_train, X_test, y_test, num_classes = load.loadDataset("mnist")
    
    #create model from textfile
    model = cMod.createModel(filename=file_path)
    
    #set number of epochs
    epochs = 50
    
    #print summary
    print(model.summary())
    
    #fit model and print results
    model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = epochs, batch_size = 200)
    score = model.evaluate(X_test, y_test, verbose = 0)
    print("Accuracy: %.2f%%" % (score[1] * 100))
    
    # save the model
    model.save(model_name)
    

run_model()