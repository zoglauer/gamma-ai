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
BatchSize = 128

MaxEvents = 2000



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

TestingTrainingSplit = 0.8

OutputDirectory = "Results"


parser = argparse.ArgumentParser(description='Perform training and/or testing of the pair identification machine learning tools.')
parser.add_argument('-f', '--filename', default='PairIdentification.inc1.id1.sim.gz', help='File name used for training/testing')
parser.add_argument('-m', '--maxevents', default='1000', help='Maximum number of events to use')
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

        if NumberOfDataSets > 0 and NumberOfDataSets % 100 == 0:
            print("Data sets processed: {}".format(NumberOfDataSets))

print("Info: Parsed {} events".format(NumberOfDataSets))

# Split the data sets in training and testing data sets

TestingTrainingSplit = 0.75

numEvents = len(DataSets)

numTraining = int(numEvents * TestingTrainingSplit)

TrainingDataSets = DataSets[:numTraining]
TestingDataSets = DataSets[numTraining:]

ValidationDataSets = TestingDataSets

# For validation/test split
# ValidationDataSets = DataSets[:int(len(TestingDataSets)/2)]
# TestingDataSets = TestingDataSets[int(len(TestingDataSets)/2):]



print("###### Data Split ########")
print("Training/Testing Split: {}".format(TestingTrainingSplit))
print("Total Data: {}, Training Data: {}, Testing Data: {}".format(numEvents, len(TrainingDataSets), len(TestingDataSets)))
print("##########################")


###################################################################################################
# Step 5: Training and evaluating the network
###################################################################################################

#TODO: Implement total energy as a feature; if no performance still poor attempt multiple models based on energy level


print("Initializing Tensors...")


BatchSize = 128

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
testing_generator = tensor_generator(TestingDataSets, BatchSize)


train_list = list(training_generator)
validation_list = list(validation_generator)


terms = 10
pos_vectors = train_list[0][0][0]
gamma_vectors = train_list[0][0][1]
label_vectors = train_list[0][1]
zero = np.zeros(shape=(1,))
ratios = []

for num in range(terms):
    print("\nTensor Number: {} \n".format(num))
    numE = 0
    for i in range(XBins):
        for j in range(YBins):
            for k in range(ZBins):
                if pos_vectors[num][i][j][k][0] != zero:
                    print("Hit Location: {}, {}, {}; Energy: {}".format(i, j, k, pos_vectors[num][i][j][k][0]))
                    numE += 1
    print("Total Nonzero Events: {}".format(numE))
    for i in range(OutputDataSpaceSize):
        if label_vectors[num][i] != zero:
            print("\nOutput marked as {}\n".format(i))
    print("Gamma Energy: {}\n".format(gamma_vectors[num][0]))
    print("Corresponding Event:")
    TrainingDataSets[num].print()
    ratios.append(numE/len(TrainingDataSets[num].E))

print("\nAverage Non-Zero Inputs per Event Hit: {}".format(sum(ratios)/len(ratios)))












# numBatches = int(len(TrainingDataSets)/BatchSize)
#Elements are each one batch with [In, Out]--for running in batches
#Elements are each one batch with [In, Out]--for running in batches
# tensors = []
# energy_tensors = []
# for i in range(numBatches):
#     if i % 100 == 0 and i > 0:
#         print("Created {} tensors".format(i))
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
# test_tensors = []
# test_energy_tensors = []
#
# for i in range(int(len(TestingDataSets)/BatchSize)):
#     if i % 100 == 0 and i > 0:
#         print("Created {} tensors".format(i))
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

# if len(tensors) != len(test_energy_tensors):
#     print("ERROR two training inputs not of same size")

# if len(test_tensors) != len(test_energy_tensors):
#     print("ERROR two test inputs not of same size")
#
#
# inT = tensors[0][0]
# outT = tensors[0][1]
# num = 0
# numT = 1
# ratios = []
# zeroes = inT[0][0][0][0][0]
#
# print("\nVisualizing Tensors:\n")
# for item in inT[:numT]:
#
#     print("\nTensor Number: {} \n".format(num))
#     numE = 0
#     for i in range(XBins):
#         for j in range(YBins):
#             for k in range(ZBins):
#                 if item[i][j][k][0] != item[0][0][0][0]:
#                     print("Hit Location: {}, {}, {}; Energy: {}".format(i, j, k, item[i][j][k][0]))
#                     numE += 1
#     print("Corresponding Gamma Energy: {}".format(energy_tensors[0][0][num][0]))
#     print("{} Non-Zero Inputs\n \nOriginal Event: \n".format(numE))
#     TrainingDataSets[num].print()
#     ratios.append(numE/len(TrainingDataSets[num].E))
#     num += 1
#
#
# print("\nAverage Non-Zero Inputs per Event Hit: {}".format(sum(ratios)/len(ratios)))
#
