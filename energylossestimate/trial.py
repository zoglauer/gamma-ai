


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


# Step 1: Input parameters
###################################################################################################


# Default parameters

# X, Y, Z bins
XBins = 32
YBins = 32
ZBins = 64

# File names
FileName = "EnergyLossElectrons.inc1.id1.data"

# Depends on GPU memory and layout
BatchSize = 128

# Split between training and testing data
TestingTrainingSplit = 0.1

MaxEvents = 100000


# Determine derived parameters

OutputDataSpaceSize = 1

XMin = -43
XMax = 43


3

# XMin = -5
# XMax = +5

YMin = -43
YMax = 43

# YMin = -5
# YMax = +5

ZMin = 13
ZMax = 45

OutputDirectory = "Results"

parser = argparse.ArgumentParser(description='Perform training and/or testing of the event clustering machine learning tools.')
parser.add_argument('-f', '--filename', default='Sim_2MeV_5GeV_flat.source', help='File name used for training/testing')
parser.add_argument('-m', '--maxevents', default='10000', help='Maximum number of events to use')
parser.add_argument('-s', '--testingtrainingsplit', default='0.1', help='Testing-training split')
parser.add_argument('-b', '--batchsize', default='128', help='Batch size')

args = parser.parse_args()

if args.filename != "":
  FileName = args.filename

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
  print("You pressed Ctrl+C - waiting for graceful abort, or press Ctrl-C again, for quick exit.")
signal.signal(signal.SIGINT, signal_handler)


# Everything ROOT related can only be loaded here otherwise it interferes with the argparse
from EventData import EventData # TODO: assign this line to relevant class.




###################################################################################################
# Step 3: Read the data
###################################################################################################


print("\n\nStarted reading data sets")
## TODO: merge old data load code with this code.

with open(FileName, "rb") as FileHandle:
  DataSets = pickle.load(FileHandle)

NumberOfDataSets = len(DataSets)

print("Info: Parsed {} events".format(NumberOfDataSets))

## FROM PREVIOUS ENERGYLOSSESTIMATE.PY
'''
  def loadData(self):
    """
    Prepare numpy array datasets for scikit-learn and tensorflow models
    
    Returns:
      list: list of the events types in numerical form: 1x: Compton event, 2x pair event, with x the detector (0: passive material, 1: tracker, 2: absober)
      list: list of all hits as a numpy array containing (x, y, z, energy) as row 
    """
   
    print("{}: Load data from sim file".format(time.time()))


    import ROOT as M

    # Load MEGAlib into ROOT
    M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")

    # Initialize MEGAlib
    G = M.MGlobal()
    G.Initialize()
    
    # Fixed for the time being
    GeometryName = "$(MEGALIB)/resource/examples/geomega/GRIPS/GRIPS_extended.geo.setup"

    # Load geometry:
    Geometry = M.MDGeometryQuest()
    if Geometry.ScanSetupFile(M.MString(GeometryName)) == True:
      print("Geometry " + GeometryName + " loaded!")
    else:
      print("Unable to load geometry " + GeometryName + " - Aborting!")
      quit()
    

    Reader = M.MFileEventsSim(Geometry)
    if Reader.Open(M.MString(self.FileName)) == False:
      print("Unable to open file " + FileName + ". Aborting!")
      quit()

    #Hist = M.TH2D("Energy", "Energy", 100, 0, 600, 100, 0, 600)
    #Hist.SetXTitle("Input energy [keV]")
    #Hist.SetYTitle("Measured energy [keV]")


    EventTypes = []
    EventHits = []
    EventEnergies = []
    GammaEnergies = []
    PairEvents = []

    NEvents = 0
    while True: 
      print("   > {} Events Processed...".format(NEvents), end='\r')

      Event = Reader.GetNextEvent()
      if not Event:
        break
  
      Type = 0
      if Event.GetNIAs() > 0:
        #Second IA is "PAIR" (GetProcess) in detector 1 (GetDetectorType()
        GammaEnergies.append(Event.GetIAAt(0).GetSecondaryEnergy())
        if Event.GetIAAt(1).GetProcess() == M.MString("COMP"):
          Type += 0 + Event.GetIAAt(1).GetDetectorType()
        elif Event.GetIAAt(1).GetProcess() == M.MString("PAIR"):
          Type += 10 + Event.GetIAAt(1).GetDetectorType()
      else:
        break
      
      if Type+1 > self.MaxLabel:
        self.MaxLabel = Type +1
  
      TotalEnergy = 0
      Hits = np.zeros((Event.GetNHTs(), 4))
      for i in range(0, Event.GetNHTs()):
        Hits[i, 0] = Event.GetHTAt(i).GetPosition().X()
        Hits[i, 1] = Event.GetHTAt(i).GetPosition().Y()
        Hits[i, 2] = Event.GetHTAt(i).GetPosition().Z()
        hitEnergy = Event.GetHTAt(i).GetEnergy()
        Hits[i, 3] = hitEnergy
        TotalEnergy += hitEnergy
      
      NEvents += 1
      EventTypes.append(Type)
      EventHits.append(Hits)
      EventEnergies.append(TotalEnergy)
      
      if NEvents >= self.MaxEvents:
        break
  
    print("Occurances of different event types:")
    print(collections.Counter(EventTypes))
    
    import math

    self.LastEventIndex = 0
    self.EventHits = EventHits
    self.EventTypes = EventTypes 
    self.EventEnergies = EventEnergies
    self.GammaEnergies = GammaEnergies

    with open('EventEnergies.data', 'wb') as filehandle:
      pickle.dump(self.EventEnergies, filehandle)
    with open('GammaEnergies.data', 'wb') as filehandle:
      pickle.dump(self.GammaEnergies, filehandle)
     
    ceil = math.ceil(len(self.EventHits)*0.75)
    self.EventTypesTrain = self.EventTypes[:ceil]
    self.EventTypesTest = self.EventTypes[ceil:]
    self.EventHitsTrain = self.EventHits[:ceil]
    self.EventHitsTest = self.EventHits[ceil:]
    self.EventEnergiesTrain = self.EventEnergies[:ceil]
    self.EventEnergiesTest = self.EventEnergies[ceil:]

    self.NEvents = NEvents

    self.DataLoaded = True

    return 
  
  def getEnergies(self):
    if os.path.exists('EventEnergies.data') and os.path.exists('GammaEnergies.data'):
      with open('EventEnergies.data', 'rb') as filehandle:
        EventEnergies = pickle.load(filehandle)
      with open('GammaEnergies.data', 'rb') as filehandle:
        GammaEnergies = pickle.load(filehandle)
      print(len(EventEnergies), len(GammaEnergies))
      if len(EventEnergies) == len(GammaEnergies) >= self.MaxEvents:
        return EventEnergies[:self.MaxEvents]
'''

###################################################################################################
# Step 4: Split the data into training, test & verification data sets
###################################################################################################


# Split the data sets in training and testing data sets

# The number of available batches in the inoput data
NBatches = int(len(DataSets) / BatchSize)


###################################################################################################
# Step 5: Setting up the neural network
###################################################################################################


print("Info: Setting up neural network...")


# Basic Voxnet Model with dense layer output (size 1)
def model1(self):
  Model = models.Sequential()
  Model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(XBins, YBins, ZBins, 1)))
  Model.add(layers.MaxPooling3D((2, 2, 3)))
  Model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
  Model.add(layers.MaxPooling3D((2, 2, 2)))
  Model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))

  Model.add(layers.Flatten())
  Model.add(layers.Dense(64, activation='relu'))
  Model.add(layers.Dense(OutputDataSpaceSize))
      
  Model.compile(optimizer=tf.keras.optimizers.Adam(epsilon=1e-08), loss=tf.keras.losses.MeanAbsoluteError(), metrics=['mae'])
      
  Model.summary()

  self.dataLoader.keras_model = Model

'''
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3D 
tf.keras.layers.Conv3D(
    filters, kernel_size, strides=(1, 1, 1), padding='valid',
    data_format=None, dilation_rate=(1, 1, 1), groups=1, activation=None,
    use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, **kwargs
)
'''
def model2(self):

    model = models.Sequential()
    #conv_1 = layers.Conv3D(32, 5, 2, 'valid', activation='relu', input_shape=XBins, YBins, ZBins, 1)
    # batch size, kernel_size, activation, ...,
    model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=XBins, YBins, ZBins, 1))  # TODO: make sure this aligns with new batch sizes.
    #batch_1 
    model.add(layers.BatchNormalization())
    #leaky relu
    model.add(layers.LeakyReLU(alpha = 0.1))

    #conv_2 = layers.Conv3D(32, 3, 1, 'valid')
    model.add(layers.Conv3d(32,3,1, activation='relu'))
    #batch_2 = layers.BatchNormalization()(conv_2)
    model.add(layers.BatchNormalization())
    #leaky relu 2 = layers.LeakyReLU(alpha = 0.1)(batch_2)
    model.add(layers.LeakyReLU(alpha=.1))
    
    #max_pool_3d = layers.MaxPooling3D(pool_size = (2,2,2), strides = 2)(max_2)
    model.add(layers.MaxPooling3D(pool_size=(2,2,2), strides=2))
    #reshape = layers.Flatten()(max_pool_3d)
    model.add(layers.Flatten())

    #dense_1 = layers.Dense(64)(reshape)
    model.add(layers.Dense(64, activation='relu')) #TODO: Figure out if there is a batchnorm relu setting!

    #dense_2 = layers.Dense(64)(drop)
    print(".... output layer ....")
    model.add(layers.Dense(OutputDataSpaceSize)) # output size should be default 1 (see section 1)

    model.compile(optimizer=tf.keras.optimizers.Adam(epsilon=1e-08), loss=tf.keras.losses.MeanAbsoluteError(), metrics=['mae'])
        
    model.summary()
    
    
    self.dataLoader.keras_model = model



###################################################################################################
# Step 6: Training and evaluating the network
###################################################################################################


print("Info: Training and evaluating the network")

# Train the network
BestLoss = sys.float_info.max
IterationOutputInterval = 10
CheckPointNum = 0

## TODO:: create training and validation looping.