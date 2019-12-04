###################################################################################################
#
# PairIdentification.py
#
# Copyright (C) by Andreas Zoglauer & Harrison Costatino.
#
# Please see the file LICENSE in the main repository for the copyright-notice.
#
###################################################################################################



###################################################################################################


import tensorflow as tf
import numpy as np

#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

import random

import signal
import sys
import time
import math
import csv
import os
import argparse
from datetime import datetime
from functools import reduce


print("\nPair Identification")
print("============================\n")



# Step 1: Input parameters
###################################################################################################


# Default parameters
# X, Y, Z bins
XBins = 32
YBins = 32
ZBins = 64

# Depends on GPU memory and layout
BatchSize = 128

MaxEvents = 100000



# Determine derived parameters

OutputDataSpaceSize = ZBins

XMin = -43
XMax = 43

# XMin = -5
# XMax = +5

YMin = -43
YMax = 43

# YMin = -5
# YMax = +5

ZMin = 13
ZMax = 45



###################################################################################################
# Step 4: Setting up the neural network
###################################################################################################

#TODO: Tweak/optimize model
# Is there a better loss function?
#Make more efficient for larger data sets


print("Info: Setting up neural network...")

print("Info: Setting up 3D CNN...")
conv_model = tf.keras.models.Sequential(name='Pair Identification CNN')
conv_model.add(tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=2, input_shape=(XBins, YBins, ZBins, 1)))
# conv_model.add(tf.keras.layers.MaxPooling3D((2,2,1)))
conv_model.add(tf.keras.layers.LeakyReLU(alpha=0.25))
conv_model.add(tf.keras.layers.BatchNormalization())
conv_model.add(tf.keras.layers.Conv3D(filters=96, kernel_size=3, strides=1, activation='relu'))
conv_model.add(tf.keras.layers.BatchNormalization())
# conv_model.add(tf.keras.layers.MaxPooling3D((2,2,1)))
conv_model.add(tf.keras.layers.Flatten())
conv_model.add(tf.keras.layers.Dense(3*OutputDataSpaceSize, activation='relu'))
conv_model.add(tf.keras.layers.BatchNormalization())
print("Conv Model Summary: ")
print(conv_model.summary())




print("Info: Setting up Numerical/Categorical Data...")
base_model = tf.keras.models.Sequential(name='Base Model')
base_model.add(tf.keras.layers.Dense(3*OutputDataSpaceSize, activation='relu', input_shape=(1,)))
base_model.add(tf.keras.layers.BatchNormalization())
print("Base Model Summary: ")
print(base_model.summary())


print("Info: Setting up Combined NN...")
combinedInput = tf.keras.layers.concatenate([conv_model.output, base_model.output])
combinedLayer = tf.keras.layers.Dense(OutputDataSpaceSize, activation='softmax')(combinedInput)
combined_model = tf.keras.models.Model([conv_model.input, base_model.input], combinedLayer)
print("Combined Model Summary: ")
print(combined_model.summary())
