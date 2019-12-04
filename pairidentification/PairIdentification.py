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

# File names
FileName = "PairIdentification.p1.sim.gz"
GeometryName = "$(MEGALIB)/resource/examples/geomega/GRIPS/GRIPS.geo.setup"

# Depends on GPU memory and layout

#Set later
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

#Set in stone later
TestingTrainingSplit = 0.8

OutputDirectory = "Results"


parser = argparse.ArgumentParser(description='Perform training and/or testing of the pair identification machine learning tools.')
parser.add_argument('-f', '--filename', default='PairIdentification.p1.sim.gz', help='File name used for training/testing')
parser.add_argument('-m', '--maxevents', default='10000', help='Maximum number of events to use')
parser.add_argument('-s', '--testingtrainigsplit', default='0.1', help='Testing-training split')
parser.add_argument('-b', '--batchsize', default='128', help='Batch size')

args = parser.parse_args()

if args.filename != "":
  FileName = args.filename

if int(args.maxevents) > 1000:
  MaxEvents = int(args.maxevents)

if int(args.batchsize) >= 16:
  BatchSize = int(args.batchsize)

if float(args.testingtrainigsplit) >= 0.05:
  TestingTrainingSplit = float(args.testingtrainigsplit)


if os.path.exists(OutputDirectory):
  Now = datetime.now()
  OutputDirectory += Now.strftime("_%Y%m%d_%H%M%S")

os.makedirs(OutputDirectory)



###################################################################################################
# Step 2: Global functions
###################################################################################################


# Take care of Ctrl-C
Interrupted = False
NInterrupts = 0
def signal_handler(signal, frame):
  global Interrupted
  Interrupted = True
  global NInterrupts
  NInterrupts += 1
  if NInterrupts >= 2:
    print("Aborting!")
    sys.exit(0)
  print("You pressed Ctrl+C - waiting for graceful abort, or press  Ctrl-C again, for quick exit.")
signal.signal(signal.SIGINT, signal_handler)


# Everything ROOT related can only be loaded here otherwise it interferes with the argparse
from EventData import EventData

# Load MEGAlib into ROOT so that it is usable
import ROOT as M
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")
M.PyConfig.IgnoreCommandLineOptions = True



###################################################################################################
# Step 3: Create some training, test & verification data sets
###################################################################################################


# Read the simulation file data:
DataSets = []


# Load geometry:
Geometry = M.MDGeometryQuest()
if Geometry.ScanSetupFile(M.MString(GeometryName)) == True:
  print("Geometry " + GeometryName + " loaded!")
else:
  print("Unable to load geometry " + GeometryName + " - Aborting!")
  quit()


Reader = M.MFileEventsSim(Geometry)
if Reader.Open(M.MString(FileName)) == False:
  print("Unable to open file " + FileName + ". Aborting!")
  quit()


print("\n\nStarted reading data sets")
NumberOfDataSets = 0
while NumberOfDataSets < MaxEvents:
  Event = Reader.GetNextEvent()
  if not Event:
    break

  if Event.GetNIAs() > 0:
    Data = EventData()
    if Data.parse(Event) == True:
      if Data.hasHitsOutside(XMin, XMax, YMin, YMax, ZMin, ZMax) == False:
        DataSets.append(Data)
        NumberOfDataSets += 1
        if NumberOfDataSets % 500 == 0:
            print("Data sets processed: {}".format(NumberOfDataSets))

print("Info: Parsed {} events".format(NumberOfDataSets))

# Split the data sets in training and testing data sets

TestingTrainingSplit = 0.75


numEvents = len(DataSets)

numTraining = int(numEvents * TestingTrainingSplit)

TrainingDataSets = DataSets[:numTraining]
ValidationDataSets = DataSets[numTraining:]



# For testing/validation split
# ValidationDataSets = TestingDataSets[:int(len(TestingDataSets)/2)]
# TestingDataSets = TestingDataSets[int(len(TestingDataSets)/2):]

print("###### Data Split ########")
print("Training/Testing Split: {}".format(TestingTrainingSplit))
print("Total Data: {}, Training Data: {}, Validation Data: {},Testing Data: {}".format(numEvents, len(TrainingDataSets), len(ValidationDataSets), len(TestingDataSets)))
print("##########################")


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




print("Info: Setting up Gamma Energy Model...")
base_model = tf.keras.models.Sequential(name='Gamma Model')
base_model.add(tf.keras.layers.Dense(int(1.5*OutputDataSpaceSize), activation='relu', input_shape=(1,)))
base_model.add(tf.keras.layers.BatchNormalization())
print("Base Model Summary: ")
print(base_model.summary())


print("Info: Setting up Combined NN...")
combinedInput = tf.keras.layers.concatenate([conv_model.output, base_model.output])
combinedLayer = tf.keras.layers.Dense(OutputDataSpaceSize, activation='softmax')(combinedInput)
combined_model = tf.keras.models.Model([conv_model.input, base_model.input], combinedLayer)
print("Combined Model Summary: ")
print(combined_model.summary())



combined_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])




# Session configuration
print("      ... configuration ...")
Config = tf.ConfigProto()
Config.gpu_options.allow_growth = True

# Create and initialize the session
print("      ... session ...")
Session = tf.Session(config=Config)
Session.run(tf.global_variables_initializer())

print("      ... listing uninitialized variables if there are any ...")
print(tf.report_uninitialized_variables())


print("      ... writer ...")
writer = tf.summary.FileWriter(OutputDirectory, Session.graph)
writer.close()

# Add ops to save and restore all the variables.
print("      ... saver ...")
Saver = tf.train.Saver()

K = tf.keras.backend
K.set_session(Session)


###################################################################################################
# Step 5: Training and evaluating the network
###################################################################################################

#TODO: attempt multiple models based on energy level
#TODO: Try a dual model setup for high and low energy setups: 10-20 and then 21+
# TODO: Add more robust model performance evaluation
#TODO: Modularize tensor set up and try sequence for training

BatchSize = 128


print("Initializing Tensors...")

class tensor_generator(tf.keras.utils.Sequence):

    def __init__(self, event_data, batch_size):
        self.event_data = event_data
        self.batch_size = batch_size

    def __len__(self):
        return int(len(self.event_data)/self.batch_size)

    def __getitem__(self, idx):
        pos_tensor = make_positional_tensor(self.event_data, idx, self.batch_size)
        gamma_tensor = make_gamma_tensor(self.event_data, idx, self.batch_size)
        label_tensor = make_label_tensor(self.event_data, idx, self.batch_size)

        if len(pos_tensor) != len(gamma_tensor) and len(pos_tensor) != len(label_tensor):
            raise Exception("Bad tensor sizes")

        return ([pos_tensor, gamma_tensor], label_tensor)




def make_positional_tensor(event_data, idx, batch_size):
    tensor = np.zeros(shape=(batch_size, XBins, YBins, ZBins, 1))
    for i in range(batch_size):
        Event = event_data[i + idx*batch_size]
        for j in range(len(Event.X)):
            XBin = int( (Event.X[j] - XMin) / ((XMax - XMin) / XBins) )
            YBin = int( (Event.Y[j] - YMin) / ((YMax - YMin) / YBins) )
            ZBin = int( (Event.Z[j] - ZMin) / ((ZMax - ZMin) / ZBins) )
            if XBin >= 0 and YBin >= 0 and ZBin >= 0 and XBin < XBins and YBin < YBins and ZBin < ZBins:
                tensor[i][XBin][YBin][ZBin][0] = Event.E[j]
    return tensor




def make_gamma_tensor(event_data, idx, batch_size):
    tensor = np.zeros(shape=(batch_size, 1))
    for i in range(batch_size):
        event = event_data[i + idx*batch_size]
        tensor[i][0] = event.GammaEnergy
    return tensor


def make_label_tensor(event_data, idx, batch_size):
    tensor = np.zeros(shape=(batch_size, OutputDataSpaceSize))
    for i in range(batch_size):
        Event = event_data[i + idx*BatchSize]
        # Set the layer in which the event happened
        if Event.OriginPositionZ > ZMin and Event.OriginPositionZ < ZMax:
            LayerBin = int ((Event.OriginPositionZ - ZMin) / ((ZMax- ZMin)/ ZBins) )
            tensor[i][LayerBin] = 1
        else:
            #TODO May need to reevaluate this line
            tensor[i][OutputDataSpaceSize-1] = 1
    return tensor




training_generator = tensor_generator(TrainingDataSets, BatchSize)
validation_generator = tensor_generator(ValidationDataSets, BatchSize)
# testing_generator = tensor_generator(TestingDataSets, BatchSize)


print("Training Model...")

history = combined_model.fit_generator(generator=training_generator, verbose=1, epochs=50, validation_data=validation_generator, shuffle=True)

print("Finished Training\nHistory is: \n")
print(history.history)

print("Evaluating Model...")

loss, acc = combined_model.evaluate_generator(generator=validation_generator, verbose=1)

print("Final Model has accuracy: {} with loss: {}".format(acc, loss))



#Non generated way of setting up data
# numBatches = int(len(TrainingDataSets)/BatchSize)
#
# #Elements are each one batch with [In, Out]--for running in batches
# tensors = []
# energy_tensors = []
#
# for i in range(numBatches):
#     if i % 10 == 0 and i > 0:
#         print("Created {} train tensors".format(i))
#
#     InputTensor = np.zeros(shape=(BatchSize, XBins, YBins, ZBins, 1))
#     OutputTensor = np.zeros(shape=(BatchSize, OutputDataSpaceSize))
#     InputEnergyTensor = np.zeros(shape=(BatchSize, 1))
#
#     for j in range(BatchSize):
#         Event = TrainingDataSets[j + i*BatchSize]
#         # Set the layer in which the event happened
#         if Event.OriginPositionZ > ZMin and Event.OriginPositionZ < ZMax:
#             LayerBin = int ((Event.OriginPositionZ - ZMin) / ((ZMax- ZMin)/ ZBins) )
#             OutputTensor[j][LayerBin] = 1
#         else:
#             #May need to reevaluate this line
#             OutputTensor[j][OutputDataSpaceSize-1] = 1
#
#       # Set all the hit locations and energies
#         for k in range(len(Event.X)):
#             XBin = int( (Event.X[k] - XMin) / ((XMax - XMin) / XBins) )
#             YBin = int( (Event.Y[k] - YMin) / ((YMax - YMin) / YBins) )
#             ZBin = int( (Event.Z[k] - ZMin) / ((ZMax - ZMin) / ZBins) )
#             if XBin >= 0 and YBin >= 0 and ZBin >= 0 and XBin < XBins and YBin < YBins and ZBin < ZBins:
#                 InputTensor[j][XBin][YBin][ZBin][0] = Event.E[k]
#
#         InputEnergyTensor[j][0] = Event.GammaEnergy
#
#     tensors.append([InputTensor, OutputTensor])
#     energy_tensors.append([InputEnergyTensor, OutputTensor])
#
#
# test_tensors = []
# test_energy_tensors = []
#
# for i in range(int(len(TestingDataSets)/BatchSize)):
#     if i % 10 == 0 and i > 0:
#         print("Created {} test tensors".format(i))
#
#     InputTensor = np.zeros(shape=(BatchSize, XBins, YBins, ZBins, 1))
#     OutputTensor = np.zeros(shape=(BatchSize, OutputDataSpaceSize))
#     InputEnergyTensor = np.zeros(shape=(BatchSize, 1))
#
#     for j in range(BatchSize):
#         Event = TestingDataSets[j + i*BatchSize]
#         # Set the layer in which the event happened
#         if Event.OriginPositionZ > ZMin and Event.OriginPositionZ < ZMax:
#             LayerBin = int ((Event.OriginPositionZ - ZMin) / ((ZMax- ZMin)/ ZBins) )
#             OutputTensor[j][LayerBin] = 1
#         else:
#             OutputTensor[j][OutputDataSpaceSize-1] = 1
#
#       # Set all the hit locations and energies
#         for k in range(len(Event.X)):
#             XBin = int( (Event.X[k] - XMin) / ((XMax - XMin) / XBins) )
#             YBin = int( (Event.Y[k] - YMin) / ((YMax - YMin) / YBins) )
#             ZBin = int( (Event.Z[k] - ZMin) / ((ZMax - ZMin) / ZBins) )
#             if XBin >= 0 and YBin >= 0 and ZBin >= 0 and XBin < XBins and YBin < YBins and ZBin < ZBins:
#                 InputTensor[j][XBin][YBin][ZBin][0] = Event.E[k]
#
#         InputEnergyTensor[j][0] = Event.GammaEnergy
#
#     test_tensors.append([InputTensor, OutputTensor])
#     test_energy_tensors.append([InputEnergyTensor, OutputTensor])
#
# if len(tensors) != len(energy_tensors):
#     print("ERROR two training inputs not of same size")
#
# if len(test_tensors) != len(test_energy_tensors):
#     print("ERROR two test inputs not of same size")
#
# print("Training Model...")
# history = []
# for i in range(len(tensors)):
#     history.append(combined_model.fit([tensors[i][0], energy_tensors[i][0]], tensors[i][1], epochs=5))
#
#
#
# print("Checking Performance...")
# acc_list = []
# loss_list = []
#
# accTestlist = []
# lossTestlist = []
#
# for i in range(len(tensors)):
#     loss, acc = combined_model.evaluate([tensors[i][0], energy_tensors[i][0]], tensors[i][1])
#     accTestlist.append(acc)
#     lossTestlist.append(loss)
#
# for i in range(len(test_tensors)):
#     loss, acc = combined_model.evaluate([test_tensors[i][0], test_energy_tensors[i][0]], test_tensors[i][1])
#     acc_list.append(acc)
#     loss_list.append(loss)
#
# for i in range(len(accTestlist)):
#     print('On training round {} the accuracy was {} with loss {}'.format(i, accTestlist[i], lossTestlist[i]))
#
# for i in range(len(acc_list)):
#     print('On test round {} the accuracy was {} with loss {}'.format(i, acc_list[i], loss_list[i]))
