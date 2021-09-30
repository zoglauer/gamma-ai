###################################################################################################
#
# CreateIdealData.py
#
# Copyright (C) by Andreas Zoglauer & contributors.
# All rights reserved.
#
# Please see the file LICENSE in the main repository for the copyright-notice.
#
###################################################################################################



###################################################################################################



# Base Python
import signal
import sys
import pickle
import random
import os
import argparse


import numpy as np

# Everything ROOT related can only be loaded here otherwise it interferes with the argparse
from EventData import EventData




print("\nCreate Ideal data")
print("============================\n")



# Step 1: Input parameters
###################################################################################################


# Default parameters


parser = argparse.ArgumentParser(description='Extract events from Cosima sim file.')
parser.add_argument('-f', '--filename', default='RecoilElectrons.ideal.data', help='Output file name')
parser.add_argument('-m', '--maxdatasets', default=10000, type=int, help='Maximum number of good data sets to extract')

args = parser.parse_args()

FileName = args.filename

MaxDataSets = args.maxdatasets

OutputFileName = FileName
if not OutputFileName.endswith(".data"):
  OutputFileName += ".data"


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




###################################################################################################
# Step 3: Create the data set
###################################################################################################



###################################################################################################



# Read the simulation file data:
DataSets = []
for i in range(0, MaxDataSets):
  Data = EventData()
  Data.ID = i
  
  x = random.uniform(-43, 43)
  y = random.uniform(-43, 43)
  z = random.uniform(13, 45)
  
  Data.TrackRealStartX = x
  Data.TrackRealStartY = y
  Data.TrackRealStartZ = z

  Data.TrackRealDirectionX = 0.0
  Data.TrackRealDirectionY = 0.0
  Data.TrackRealDirectionZ = 1.0

  Data.TrackMeasuredStartX = x
  Data.TrackMeasuredStartY = y
  Data.TrackMeasuredStartZ = z

  Data.TrackMeasuredDirectionX = 0.0
  Data.TrackMeasuredDirectionY = 0.0
  Data.TrackMeasuredDirectionZ = 1.0

  Data.TrackSequence = np.zeros(shape=(1), dtype=int)
  Data.X = np.zeros(shape=(1), dtype=float)
  Data.Y = np.zeros(shape=(1), dtype=float)
  Data.Z = np.zeros(shape=(1), dtype=float)
  Data.E = np.zeros(shape=(1), dtype=float)

  Data.TrackSequence[0] = 0
  Data.X[0] = x
  Data.Y[0] = y
  Data.Z[0] = z
  Data.E[0] = 1000

  DataSets.append(Data)


  if Interrupted == True:
    break

print("Info: Created {} events".format(len(DataSets)))


###################################################################################################
# Step 4: Store the data
###################################################################################################



# store the data as binary data stream
print("Info: Storing the data")
with open(OutputFileName, "wb") as FileHandle:
  pickle.dump(DataSets, FileHandle)
print("Info: Done")

#input("Press [enter] to EXIT")
sys.exit(0)


###################################################################################################
