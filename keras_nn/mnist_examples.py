# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:26:03 2018

@author: Julia Bristow
"""

# import MNIST digit dataset
from keras.datasets import mnist
import matplotlib.pyplot as plt

# load the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# plot 4 images as a gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()