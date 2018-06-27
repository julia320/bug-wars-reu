# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 09:45:31 2018

@author: Julia
"""

from keras.datasets import mnist, cifar10, cifar100


dataset = input("What dataset do you want to use? Enter \'mnist\', \'cifar10\', or \'cifar100\': ")
if dataset == 'mnist':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
elif dataset == 'cifar10':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
elif dataset == 'cifar100':
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
else:
    print("That was not a valid dataset. MNIST will be used as default.")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()