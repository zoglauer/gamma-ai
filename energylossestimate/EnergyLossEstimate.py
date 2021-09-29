import tensorflow as tf
from tensorflow.keras import datasets, layers, models


import numpy as np

#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

import random
import pickle

import signal
import sys
import time
import math
import csv
import os
import argparse
from datetime import datetime
from functools import reduce

print("\nEnergy Loss Estimation")
print("============================\n")

####parameter input

XBins = 110
YBins = 110
ZBins = 48

##derived parameters
#might have to tune these values
XMin = -55
XMax = 55

YMin = -55
YMax = 55

ZMin = 0
ZMax = 48

#switch these from argparse to func parameters in new run.py file?
OutputDirectory = "Results"

parser = argparse.ArgumentParser(description='Perform training and/or testing of the event clustering machine learning tools.')
parser.add_argument('-f', '--filename', default='EnergyEstimate.p1.sim.gz', help='File name used for training/testing')
parser.add_argument('-m', '--maxevents', default='10000', help='Maximum number of events to use')
parser.add_argument('-s', '--testingtrainingsplit', default='0.9', help='Testing-training split')
parser.add_argument('-b', '--batchsize', default='20', help='Batch size')
parser.add_argument('-a', '--algorithm', default='voxnet', help='Algorithm') # optionality for algorithm replacement.

args = parser.parse_args()

if args.filename != "":
    FileName = args.filename#add exception for not existing

if int(args.maxevents) > 1000:
    MaxEvents = int(args.maxevents)

if int(args.batchsize) >= 16:
    BatchSize = int(args.batchsize)

if float(args.testingtrainingsplit) >= 0.05:
    TestingTrainingSplit = float(args.testingtrainingsplit)

#if os.path.exists(OutputDirectory):
#  Now = datetime.now()
#  OutputDirectory += Now.strftime("_%Y%m%d_%H%M%S")
#os.makedirs(OutputDirectory)

OutputDataSpaceSize = 1

Algorithm = voxnet_create #for now [dw about data parsing, imported w/ separate arg]

OutputDirectory = "output.txt" #vs Results?

#argparse stuff

####global funcs

#consider putting ctrl c func in run.py rewrite?
Interrupted = False
NInterrupts = 0

def signal_handler(signal, frame):
    """
    Handles Ctrl-C interrupts during runtime.
    """
    global Interrupted
    Interrupted = True
    global NInterrupts
    NInterrupts += 1
    if NInterrupts >= 2:
        print("Aborting!")
        sys.exit(0)
    print("You pressed Ctrl+C - waiting for graceful abort, or press Ctrl-C again, for quick exit.")

signal.signal(signal.SIGINT, signal_handler)

#only load ROOT related stuff here
from EventData import EventData #write EventData

####data setup
# Split the data sets in training and testing data sets

# The number of available batches in the input data
NBatches = int(len(DataSets) / BatchSize)
if NBatches < 2:
    print("Not enough data!")
    quit()

# Split the batches in training and testing according to TestingTrainingSplit
NTestingBatches = int(NBatches*TestingTrainingSplit)
if NTestingBatches == 0:
    NTestingBatches = 1
NTrainingBatches = NBatches - NTestingBatches

# Now split the actual data:
TrainingDataSets = []
for i in range(0, NTrainingBatches * BatchSize):
    TrainingDataSets.append(DataSets[i])


TestingDataSets = []
for i in range(0,NTestingBatches*BatchSize):
    TestingDataSets.append(DataSets[NTrainingBatches * BatchSize + i])


NumberOfTrainingEvents = len(TrainingDataSets)
NumberOfTestingEvents = len(TestingDataSets)

print("Info: Number of training data sets: {}   Number of testing data sets: {} (vs. input: {} and split ratio: {})".format(NumberOfTrainingEvents, NumberOfTestingEvents, len(DataSets), TestingTrainingSplit))

####neural network setup
#use dictionary to match algorithm var with functions that setup networks?
algorithm_setup = {}

def voxnet_create():
    """
    Create voxnet neural network
    """
    Model = models.Sequential()
    Model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(XBins, YBins, ZBins, 2)))
    Model.add(layers.MaxPooling3D((2, 2, 3)))
    Model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
    Model.add(layers.MaxPooling3D((2, 2, 2)))
    Model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))

    Model.add(layers.Flatten())
    Model.add(layers.Dense(64, activation='relu'))
    Model.add(layers.Dense(OutputDataSpaceSize))

    Model.compile(optimizer=tf.keras.optimizers.Adam(epsilon=1e-08), loss=tf.keras.losses.MeanAbsoluteError(), metrics=['mae'])

    return Model.summary()
def voxnet_create_batch():
    """
    Create voxnet neural network
    """
    Model = models.Sequential()
    Model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(XBins, YBins, ZBins, 2)))
    Model.add(layers.BatchNormalization())
    Model.add(layers.MaxPooling3D((2, 2, 3)))
    Model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
    Model.add(layers.BatchNormalization())
    Model.add(layers.MaxPooling3D((2, 2, 2)))
    Model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
    Model.add(layers.BatchNormalization())

    Model.add(layers.Flatten())
    Model.add(layers.Dense(64, activation='relu'))
    #Model.add(layers.BatchNormalization())
    Model.add(layers.Dense(OutputDataSpaceSize))

    Model.compile(optimizer=tf.keras.optimizers.Adam(epsilon=1e-08), loss=tf.keras.losses.MeanAbsoluteError(), metrics=['mae'])

    return Model.summary()

def voxnet_create_layer():
    """
    Create voxnet neural network
    """
    Model = models.Sequential()
    Model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(XBins, YBins, ZBins, 2)))
    Model.add(layers.LayerNormalization())
    Model.add(layers.MaxPooling3D((2, 2, 3)))
    Model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
    Model.add(layers.LayerNormalization())
    Model.add(layers.MaxPooling3D((2, 2, 2)))
    Model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
    Model.add(layers.LayerNormalization())

    Model.add(layers.Flatten())
    Model.add(layers.Dense(64, activation='relu'))
    Model.add(layers.LayerNormalization())
    Model.add(layers.Dense(OutputDataSpaceSize))

    Model.compile(optimizer=tf.keras.optimizers.Adam(epsilon=1e-08), loss=tf.keras.losses.MeanAbsoluteError(), metrics=['mae'])

    return Model.summary()


print("Info: Setting up neural network")
Model = Algorithm()

####train/eval network

print("Info: Training and evaluating the network")

# Training check vars (Bests for CheckPerformance())
BestGammaDif = sys.float_info.max

# function to check if neural network improving

def CheckPerformance():
    #pull in training check vars using global (cleaner way to do this?)
    global BestGammaDif

    Improvement = False

    TotalEvents = 0
    SumGammaDiff = 0

    # Step run all the testing batches, and determine the percentage of correct identifications
    # Step 1: Loop over all Testing batches
    for Batch in range(0, NTestingBatches):

        # Step 1.1: Convert data set into input and output tensor
        InputTensor = np.zeros(shape=(BatchSize, XBins, YBins, ZBins, 2))
        OutputTensor = np.zeros(shape=(BatchSize, OutputDataSpaceSize))

        # Loop over all training data sets and add them to the tensor
        for g in range(0, BatchSize):
            Event = TrainingDataSets[g + Batch*BatchSize]

            for h in range(0, len(Event.startX)):
                XBin = int( (Event.startX[h] - XMin) / ((XMax - XMin) / XBins) )
                YBin = int( (Event.startY[h] - YMin) / ((YMax - YMin) / YBins) )
                ZBin = int( (Event.startZ[h] - ZMin) / ((ZMax - ZMin) / ZBins) )
                #is this next part still correct condition for if statement?
                if XBin >= 0 and YBin >= 0 and ZBin >= 0 and XBin < XBins and YBin < YBins and ZBin < ZBins:
                    InputTensor[g][XBin][YBin][ZBin][0] = Event.measured_energy
                    InputTensor[g][XBin][YBin][ZBin][1] = Event.type # Why was it [0][1] before this?

    # Step 2: Run it
    Result = Model.predict(InputTensor)

    #print(Result[e])
    #print(OutputTensor[e])

    for e in range(0, BatchSize):
        Event = TestingDataSets[e + Batch*BatchSize]

        true_gamma = Event.GammaEnergy
        predicted_gamma = Result[e][0]
        GammaDiff = abs(true_gamma - predicted_gamma)

        SumGammaDiff += GammaDiff

        TotalEvents += 1

        # debugging code
        if Batch == 0 and e < 5:
            EventID = e + Batch*BatchSize + NTrainingBatches*BatchSize
            print("\nEvent {}:".format(EventID))
            DataSets[EventID].print()

        #print("Energy: {} vs {} -> {} difference".format(true_gamma, predicted_gamma, GammaDiff))

    if TotalEvents > 0:
        if SumGammaDiff / TotalEvents < BestGammaDif:
            BestGammaDiff = SumGammaDiff / TotalEvents
        Improvement = True

    print("Status: average gamma energy difference = {}".format(SumGammaDiff / TotalEvents))

    return Improvement

# main train/eval loop

TimeConverting = 0.0
TimeTraining = 0.0
TimeTesting = 0.0

Iteration = 0
MaxIterations = 50000 #might need to change this
TimesNoImprovement = 0
MaxTimesNoImprovement = 50 #might need to change this
while Iteration < MaxIterations:
    Iteration += 1
    print("\n\nStarting iteration {}".format(Iteration))

    # Step 1: Loop over all training batches
    for Batch in range(0, NTrainingBatches):
        print("Batch {} / {}".format(Batch+1, NTrainingBatches))

        # Step 1.1: Convert the data set into the input and output tensor
        TimerConverting = time.time()

        #might this need to be dif for dif algorithms?
        InputTensor = np.zeros(shape=(BatchSize, XBins, YBins, ZBins, 2)) #np.zeros(shape=(BatchSize, XBins, YBins, ZBins, 1, 1))
        OutputTensor = np.zeros(shape=(BatchSize, OutputDataSpaceSize))

        # Loop over all training data sets and add them to the tensor
        for g in range(0, BatchSize):
            Event = TrainingDataSets[g + Batch*BatchSize]

            for h in range(0, len(Event.startX)):
                XBin = int( (Event.startX[h] - XMin) / ((XMax - XMin) / XBins) )
                YBin = int( (Event.startY[h] - YMin) / ((YMax - YMin) / YBins) )
                ZBin = int( (Event.startZ[h] - ZMin) / ((ZMax - ZMin) / ZBins) )
                #is this next part still correct condition for if statement?
                if XBin >= 0 and YBin >= 0 and ZBin >= 0 and XBin < XBins and YBin < YBins and ZBin < ZBins:
                    InputTensor[g][XBin][YBin][ZBin][0] = Event.measured_energy
                    InputTensor[g][XBin][YBin][ZBin][1] = Event.type

            outputTensor[g][0] = Event.GammaEnergy

    TimeConverting += time.time() - TimerConverting

    # Step 1.2: Perform the actual training
    TimerTraining = time.time()
    History = Model.fit(InputTensor, OutputTensor, validation_split=0.1)
    Loss = History.history['loss'][-1]
    TimeTraining += time.time() - TimerTraining

    if Interrupted == True: break

    # End for all batches

    # Step 2: Check current performance
    TimerTesting = time.time()
    print("\nCurrent loss: {}".format(Loss))
    Improvement = CheckPerformance()

    if Improvement == True:
        TimesNoImprovement = 0
        print("\nFound new best model and performance!")
    else:
        TimesNoImprovement += 1

    TimeTesting += time.time() - TimerTesting

    # Exit strategy
    if TimesNoImprovement == MaxTimesNoImprovement:
        print("\nNo improvement for {} iterations. Quitting!".format(MaxTimesNoImprovement))
        break;

    print("\n\nTotal time converting per Iteration: {} sec".format(TimeConverting/Iteration))
    print("Total time training per Iteration:   {} sec".format(TimeTraining/Iteration))
    print("Total time testing per Iteration:    {} sec".format(TimeTesting/Iteration))

    ## save best model.
    Model.save('best_model')
    ''' load model using: tf.keras.models.load_model()'''


    # Take care of Ctrl-C
    if Interrupted == True: break


# End: for all iterations

#input("Press [enter] to EXIT")
sys.exit(0)
