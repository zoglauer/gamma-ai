import tensorflow as tf
from keras import datasets, layers, models
# from tensorflow.keras import datasets, layers, models

import numpy as np

# from adiShowerProfile import shower_profile

from event_data import EventData

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

# parameter input

XBins = 64
YBins = 64
ZBins = 64

# derived parameters
# might have to tune these values
XMin = -55
XMax = 55

YMin = -55
YMax = 55

ZMin = 0
ZMax = 48


# switch these from argparse to func parameters in new run.py file?
OutputDirectory = "Results"

# Default maximum number of events
MaxEvents = 1000000

# Default batch size
BatchSize = 128

# Default testing training split
TestingTrainingSplit = 0.1

# Fitted parameters
alpha = 121.48030860669718
beta = 3.1815852691851383

# All algorithms:
AlgorithmOptions = ["voxnet_create", "voxnet_create_batch",
                    "voxnet_create_layer", "az", "mixed_input", "voxnet_test", "voxnet_new_nodes"]
SigmaOptions = [0.2, 0.4, 0.6, 0.8]

parser = argparse.ArgumentParser(
    description='Perform training and/or testing of the event clustering machine learning tools.')
parser.add_argument('-f', '--filename', default='EnergyEstimate.p1.sim.gz',
                    help='File name used for training/testing')
parser.add_argument('-m', '--maxevents', default=1,
                    help='Maximum number of events to use')
parser.add_argument('-s', '--testingtrainingsplit',
                    default=TestingTrainingSplit, help='Testing-training split')
parser.add_argument('-b', '--batchsize', default=BatchSize, help='Batch size')
# optionality for algorithm replacement.
parser.add_argument('-a', '--algorithm', default='mixed_input',
                    help='Algorithm. One of [voxnet_create, voxnet_create_batch, voxnet_create_layer, az, mixed_input, voxnet_test, voxnet_new_nodes ]')
parser.add_argument('-sig', '--sigma', default=0.2,
                    help='Sigma for Gaussian Broadening. One of [0.2,0.4,0.6,0.8 ]')
args = parser.parse_args()

if args.filename != "":
    FileName = args.filename
if not os.path.exists(FileName):
    print("Error: The training data file does not exist: {}".format(FileName))
    sys.exit(0)
print("CMD: Using file {}".format(FileName))

if int(args.maxevents) >= 1000:
    MaxEvents = int(args.maxevents)
print("CMD: Using {} as maximum event number".format(MaxEvents))

if int(args.batchsize) >= 16:
    BatchSize = int(args.batchsize)
print("CMD: Using {} as batch size".format(BatchSize))

if float(args.testingtrainingsplit) >= 0.05 and float(args.testingtrainingsplit) < 0.95:
    TestingTrainingSplit = float(args.testingtrainingsplit)
print("CMD: Using {} as testing-training split".format(TestingTrainingSplit))

Algorithm = args.algorithm
if not Algorithm in AlgorithmOptions:
    print("Error: The neural network layout must be one of [{}], and not: {}".format(
        AlgorithmOptions, Algorithm))
    sys.exit(0)
print("CMD: Using {} neural network model".format(Algorithm))

Sigma = args.sigma
if not Sigma in SigmaOptions:
    print("Error: The sigma value must be one of [{}], and not: {}".format(
        SigmaOptions, Sigma))
    sys.exit(0)
print("CMD: Using {} as sigma value".format(Sigma))


# if os.path.exists(OutputDirectory):
#  Now = datetime.now()
#  OutputDirectory += Now.strftime("_%Y%m%d_%H%M%S")
# os.makedirs(OutputDirectory)

OutputDataSpaceSize = 1

OutputDirectory = "output.txt"  # vs Results?

# argparse stuff

print("Info: Setting up neural network")

####global funcs

# consider putting ctrl c func in run.py rewrite?
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

# only load ROOT related stuff here

# data setup
# Split the data sets in training and testing data sets

with open(FileName, "rb") as FileHandle:
    DataSets = pickle.load(FileHandle)

if len(DataSets) > MaxEvents:
    DataSets = DataSets[:MaxEvents]

# The number of available batches in the input data

# TODO: figure out where we get this "DataSets" object from is it supposed to be renamed ?
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
for i in range(0, NTestingBatches*BatchSize):
    TestingDataSets.append(DataSets[NTrainingBatches * BatchSize + i])


NumberOfTrainingEvents = len(TrainingDataSets)
NumberOfTestingEvents = len(TestingDataSets)

print("Info: Number of training data sets: {}   Number of testing data sets: {} (vs. input: {} and split ratio: {})".format(
    NumberOfTrainingEvents, NumberOfTestingEvents, len(DataSets), TestingTrainingSplit))


# neural network setup
# use dictionary to match algorithm var with functions that setup networks?
algorithm_setup = {}
# showerInput =
# showerOutput =


def mixed_input():  # takes in output of shower profile, default none for now
    global Model

    vox_input = layers.Input(shape=(XBins, YBins, ZBins, 1))
    vox = layers.Conv3D(32, (3, 3, 3), activation='relu',
                        padding="SAME")(vox_input)
    vox = layers.BatchNormalization()(vox)
    vox = layers.MaxPooling3D((3, 3, 3))(vox)
    vox = layers.Conv3D(64, (3, 3, 3), activation='relu', padding="SAME")(vox)
    vox = layers.MaxPooling3D((3, 3, 3))(vox)
    vox = layers.Conv3D(128, (3, 3, 3), activation='relu', padding="SAME")(vox)
    vox = layers.Flatten()(vox)

    #vox = layers.Dense(8, activation = 'relu')(vox)
    #vox = layers.Dense(OutputDataSpaceSize)(vox)
    #Model = models.Model(inputs = vox_input, outputs = vox)

    shower_input = layers.Input(shape=(2))
    shower = layers.Flatten()(shower_input)

    result = layers.Concatenate()([vox, shower])

    #result = layers.Dense(16, activation = 'relu')(result)
    result = layers.Dense(8, activation='relu')(result)
    result = layers.Dense(OutputDataSpaceSize)(result)

    Model = models.Model(inputs=[vox_input, shower_input], outputs=result)


def az():
    global Model
    Model = models.Sequential()
    Model.add(layers.Conv3D(32, (3, 3, 3), activation='relu',
              input_shape=(XBins, YBins, ZBins, 1), padding="SAME"))
    Model.add(layers.BatchNormalization())
    Model.add(layers.MaxPooling3D((3, 3, 3)))
    Model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', padding="SAME"))
    Model.add(layers.MaxPooling3D((3, 3, 3)))
    Model.add(layers.Conv3D(128, (3, 3, 3), activation='relu', padding="SAME"))

    Model.add(layers.Flatten())
    Model.add(layers.Dense(8, activation='relu'))
    # mixed_input
    Model.add(layers.Dense(OutputDataSpaceSize))
# shower model input: measured gamma energy for each event z bin
# shower model output: single vector
# vox model input: measured gamma energy x,y,z bins
# vox model output:


# one to one mapping model
def shower_mixed():
    print("In shower model")
    Showermodel = models.Sequential()
    layer = layers.Layer()
    Showermodel.add(layer)
    return Showermodel

# voxnet model for mixed input


def vox_mixed():
    vox = models.Sequential()
    vox.add(layers.Conv3D(32, (3, 3, 3), activation='relu', padding="SAME"))
    vox.add(layers.BatchNormalization())
    vox.add(layers.MaxPooling3D((3, 3, 3)))
    vox.add(layers.Conv3D(64, (3, 3, 3), activation='relu', padding="SAME"))
    vox.add(layers.MaxPooling3D((3, 3, 3)))
    vox.add(layers.Conv3D(128, (3, 3, 3), activation='relu', padding="SAME"))
    vox.add(layers.Flatten())
    return vox


def voxnet_create():
    """
    Create voxnet neural network
    """
    global Model

    Model.add(layers.Conv3D(32, (3, 3, 3), activation='relu',
              input_shape=(XBins, YBins, ZBins, 1), padding="SAME"))
    Model.add(layers.MaxPooling3D((2, 2, 3)))
    Model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', padding="SAME"))
    Model.add(layers.MaxPooling3D((2, 2, 2)))
    Model.add(layers.Conv3D(128, (3, 3, 3), activation='relu', padding="SAME"))

    Model.add(layers.Flatten())
    Model.add(layers.Dense(8, activation='relu'))
    Model.add(layers.Dense(OutputDataSpaceSize))


def voxnet_test():
    """
    Create voxnet test network
    """
    global Model

    Model.add(layers.Conv3D(32, (3, 3, 3), activation='relu',
              input_shape=(XBins, YBins, ZBins, 1)))
    Model.add(layers.MaxPooling3D((2, 2, 3)))
    Model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
    Model.add(layers.MaxPooling3D((2, 2, 2)))
    Model.add(layers.Conv3D(128, (3, 3, 3), activation='relu'))

    Model.add(layers.Flatten())
    Model.add(layers.Dense(8, activation='relu'))
    Model.add(layers.Dense(OutputDataSpaceSize))
    print(Model.summary())
    sys.exit(0)


def voxnet_new_nodes():  # testing voxnet with different node structure
    """
    Create voxnet test network
    """
    global Model

    Model.add(layers.Conv3D(128, (3, 3, 3), activation='relu',
              input_shape=(XBins, YBins, ZBins, 1), padding="SAME"))
    Model.add(layers.MaxPooling3D((2, 2, 3)))
    Model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', padding="SAME"))
    Model.add(layers.MaxPooling3D((2, 2, 2)))
    Model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', padding="SAME"))

    Model.add(layers.Flatten())
    Model.add(layers.Dense(20, activation='relu'))
    Model.add(layers.Dense(OutputDataSpaceSize))


def voxnet_create_batch():
    """
    Create voxnet neural network
    """
    global Model

    Model = models.Sequential()
    Model.add(layers.Conv3D(32, (3, 3, 3), activation='relu',
              input_shape=(XBins, YBins, ZBins, 1)))
    Model.add(layers.BatchNormalization())
    Model.add(layers.MaxPooling3D((2, 2, 3)))
    Model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
    Model.add(layers.BatchNormalization())
    Model.add(layers.MaxPooling3D((2, 2, 2)))
    Model.add(layers.Conv3D(128, (3, 3, 3), activation='relu'))
    Model.add(layers.BatchNormalization())

    Model.add(layers.Flatten())
    Model.add(layers.Dense(8, activation='relu'))
    Model.add(layers.Dense(OutputDataSpaceSize))


def voxnet_create_layer():
    """
    Create voxnet neural network
    """
    global Model

    Model.add(layers.Conv3D(32, (3, 3, 3), activation='relu',
              input_shape=(XBins, YBins, ZBins, 1)))
    Model.add(layers.LayerNormalization())
    Model.add(layers.MaxPooling3D((2, 2, 3)))
    Model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
    Model.add(layers.LayerNormalization())
    Model.add(layers.MaxPooling3D((2, 2, 2)))
    Model.add(layers.Conv3D(128, (3, 3, 3), activation='relu'))
    Model.add(layers.LayerNormalization())

    Model.add(layers.Flatten())
    Model.add(layers.Dense(8, activation='relu'))
    Model.add(layers.Dense(OutputDataSpaceSize))


print("Info: Setting up neural network")

Model = models.Sequential()
if Algorithm == "voxnet_create_batch":
    voxnet_create_batch()
elif Algorithm == "voxnet_create_layer":
    voxnet_create_layer()
elif Algorithm == "az":
    az()
elif Algorithm == "mixed_input":
    mixed_input()
    print('setting model to mixed input')
else:
    voxnet_create()

Model.compile(optimizer=tf.keras.optimizers.Adam(epsilon=1e-08),
              loss=tf.keras.losses.MeanAbsolutePercentageError(), metrics=['mape'])
Model.summary()


# train/eval network

# gaussian broadening??
# def g_broadening(sigma):
# return random.gaussian(0,sigma)


print("Info: Training and evaluating the network")

# Training check vars (Bests for CheckPerformance())
BestGammaDif = sys.float_info.max

# function to check if neural network improving


def CheckPerformance():
    # pull in training check vars using global (cleaner way to do this?)
    global BestGammaDif

    Improvement = False

    TotalEvents = 0
    SumGammaDiff = 0

    # Step run all the testing batches, and determine the percentage of correct identifications
    # Step 1: Loop over all Testing batches
    for Batch in range(0, NTestingBatches):

        # Step 1.1: Convert data set into input and output tensor
        InputTensor = np.zeros(shape=(BatchSize, XBins, YBins, ZBins, 1))
        OutputTensor = np.zeros(shape=(BatchSize, OutputDataSpaceSize))
        InputShowerTensor = np.zeros(shape=(BatchSize, 2))

        # Loop over all training data sets and add them to the tensor
        for g in range(0, BatchSize):
            Event = TrainingDataSets[g + Batch*BatchSize]
            #xdat = build_xdat([Event])

            for h in range(0, Event.hits.shape[0]):
                XBin = int((Event.hits[h, 0] - XMin) / ((XMax - XMin) / XBins))
                YBin = int((Event.hits[h, 1] - YMin) / ((YMax - YMin) / YBins))
                ZBin = int((Event.hits[h, 2] - ZMin) / ((ZMax - ZMin) / ZBins))
                # is this next part still correct condition for if statement?
                if XBin >= 0 and YBin >= 0 and ZBin >= 0 and XBin < XBins and YBin < YBins and ZBin < ZBins:
                    InputTensor[g][XBin][YBin][ZBin][0] = Event.hits[h, 3]
                    # InputShowerTensor[g][0] = Event.measured_energy* np.random.uniform(low=0.5,high=1)   #commented out until we receive shower function
                    #x0 = np.random.uniform(low=0.5,high=1)
                    #InputShowerTensor[g][0] = shower_profile(Event.hits, alpha, beta)

                    # sigma values: 0.2, 0.4, 0.6, 0.8
                    InputShowerTensor[g][0] = random.gauss(
                        Event.gamma_energy, Sigma*Event.gamma_energy)
                    InputShowerTensor[g][1] = Event.measured_energy

                else:
                    print("Warning: Hit outside grid: {}, {}, {}".format(
                        Event.hits[h, 0], Event.hits[h, 1], Event.hits[h, 2]))

        # Step 2: Run it
        if Algorithm == "mixed_input":
            Result = Model.predict([InputTensor, InputShowerTensor])
        else:
            Result = Model.predict(InputTensor)

        # print(Result[e])
        # print(OutputTensor[e])

        for e in range(0, BatchSize):
            Event = TestingDataSets[e + Batch*BatchSize]

            true_gamma = Event.gamma_energy
            predicted_gamma = Result[e][0]
            GammaDiff = abs(true_gamma - predicted_gamma) / true_gamma

            SumGammaDiff += GammaDiff

            TotalEvents += 1

            # debugging code
            if Batch == 0 and e < 10:
                EventID = e + Batch*BatchSize + NTrainingBatches*BatchSize
                print("\nEvent {}:".format(EventID))
                # DataSets[EventID].print()
                print("Energy: Input {:+.0f}, measured: {:+.0f}, predicted: {:+.0f}, difference: {:+.2f}%".format(
                    true_gamma, Event.measured_energy, predicted_gamma, 100.0 * GammaDiff))

    if TotalEvents > 0:
        if SumGammaDiff / TotalEvents < BestGammaDif:
            BestGammaDif = SumGammaDiff / TotalEvents
            Improvement = True

    print("Status: average absolute gamma energy difference: {}% vs best {}%".format(
        100.0 * SumGammaDiff / TotalEvents, 100.0 * BestGammaDif))

    return Improvement

# main train/eval loop


TimeConverting = 0.0
TimeTraining = 0.0
TimeTesting = 0.0

Iteration = 0
MaxIterations = 50000  # might need to change this
TimesNoImprovement = 0
MaxTimesNoImprovement = 100  # might need to change this
while Iteration < MaxIterations:
    Iteration += 1
    print("\n\nStarting iteration {}".format(Iteration))

    # Step 1: Loop over all training batches
    for Batch in range(0, NTrainingBatches):
        print("Batch {} / {}".format(Batch+1, NTrainingBatches))

        # Step 1.1: Convert the data set into the input and output tensor
        TimerConverting = time.time()

        # might this need to be dif for dif algorithms?
        # np.zeros(shape=(BatchSize, XBins, YBins, ZBins, 1, 1))
        InputTensor = np.zeros(shape=(BatchSize, XBins, YBins, ZBins, 1))
        OutputTensor = np.zeros(shape=(BatchSize, OutputDataSpaceSize))
        InputShowerTensor = np.zeros(shape=(BatchSize, 2))

        # Loop over all training data sets and add them to the tensor
        for g in range(0, BatchSize):
            Event = TrainingDataSets[g + Batch*BatchSize]
            #xdat = build_xdat([Event])

            for h in range(0, Event.hits.shape[0]):
                XBin = int((Event.hits[h, 0] - XMin) / ((XMax - XMin) / XBins))
                YBin = int((Event.hits[h, 1] - YMin) / ((YMax - YMin) / YBins))
                ZBin = int((Event.hits[h, 2] - ZMin) / ((ZMax - ZMin) / ZBins))
                # is this next part still correct condition for if statement?
                if XBin >= 0 and YBin >= 0 and ZBin >= 0 and XBin < XBins and YBin < YBins and ZBin < ZBins:
                    InputTensor[g][XBin][YBin][ZBin][0] = Event.hits[h, 3]
                    #print("{}, {}, {}, {}".format(XBin, YBin, ZBin, Event.hits[h, 3]))
                    # InputShowerTensor[g][0] = 0 # Event.measured_energy *np.random.uniform(low=0.5,high=1)  #commented out until we receive shower function
                    #x0 = np.random.uniform(low=0.5, high=1)
                    #InputShowerTensor[g][0] = shower_profile(Event.hits, alpha, beta)
                    # sigma values: 0.2, 0.4, 0.6, 0.8
                    InputShowerTensor[g][0] = random.gauss(
                        Event.gamma_energy, Sigma*Event.gamma_energy)
                    InputShowerTensor[g][1] = Event.measured_energy
                else:
                    print("Warning: Hit outside grid: {}, {}, {}".format(
                        Event.hits[h, 0], Event.hits[h, 1], Event.hits[h, 2]))

            OutputTensor[g][0] = Event.gamma_energy

        TimeConverting += time.time() - TimerConverting

        # Step 1.2: Perform the actual training
        TimerTraining = time.time()
        #inputData =(InputTensor, InputShowerTensor)
        #inputData = np.asarray(inputData)
        if Algorithm == "mixed_input":
            History = Model.fit(
                x=[InputTensor, InputShowerTensor], y=OutputTensor, validation_split=0.1)
        else:
            History = Model.fit(
                x=InputTensor, y=OutputTensor, validation_split=0.1)
        Loss = History.history['loss'][-1]
        TimeTraining += time.time() - TimerTraining

        if Interrupted == True:
            break

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
        print("\nNo improvement for {} iterations. Quitting!".format(
            MaxTimesNoImprovement))
        break

    print("\n\nTotal time converting per Iteration: {} sec".format(
        TimeConverting/Iteration))
    print("Total time training per Iteration:   {} sec".format(
        TimeTraining/Iteration))
    print("Total time testing per Iteration:    {} sec".format(TimeTesting/Iteration))

    # save best model.
    if Improvement == True:
        Model.save('best_model')
        ''' load model using: tf.keras.models.load_model()'''

    # Take care of Ctrl-C
    if Interrupted == True:
        break


# End: for all iterations
