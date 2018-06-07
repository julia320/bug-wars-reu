# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 15:23:27 2018

@author: Julia
"""

from keras import backend as K
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects

# activation function that returns x squared
def square_act(x):
    return x**2

get_custom_objects().update({'custom_activation': Activation(square_act)})