# configure import statements
from __future__ import absolute_import, division, print_function
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# makes tensorflow evaluate operations immediately,
# rather than creating a computational graph for later
tf.enable_eager_execution()

# print tensorflow version and whether eager execution is running
print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

# get the iris dataset 
train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

# print the iris dataset
print("Local copy of the dataset file: {}".format(train_dataset_fp))