# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:25:37 2018

@author: Julia
"""

from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects

# activation function for squaring a number
def square_activation(x):
    return x**2

# add square activation function to keras
get_custom_objects().update({'square_activation': Activation(square_activation)})