import signal
import sys
import pickle
import os
import argparse
import math

import numpy as np

# Everything ROOT related can only be loaded here otherwise it interferes with the argparse
from EventData import EventData
   
print("{}: Load data from sim file".format(time.time()))

import ROOT as M

# Load MEGAlib into ROOT:

M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")

# Initialize MEGAlib:

G = M.MGlobal()
G.Initialize()

#Handle input arguments:

parser = argparse.ArgumentParser(description='Extract events from Cosima sim file.')
parser.add_argument('-f', '--filename', default='EnergyEstimate.p1.sim.gz', help='File name used for training/testing')
parser.add_argument('-m', '--maxdatasets', default=1000000000, type=int, help='Maximum number of good data sets to extract')
parser.add_argument('-d', '--debug', default=False, action="store_true", help='Enable debugging output')
parser.add_argument('-p', '--parser', default='voxnet', help='Choose parser appropriate for algorithm taking data.')

args = parser.parse_args()

FileName = args.filename

MaxDataSets = args.maxdatasets

OutputFileName = FileName
if OutputFileName.endswith(".gz"):
    OutputFileName = OutputFileName[:-3]
if OutputFileName.endswith(".sim"):
    OutputFileName = OutputFileName[:-4]
OutputFileName += ".data"

Debug = args.debug

# Parser functions:

def parse_voxnet(SimEvent, Debug):
    """
    Extract data from SimEvent to prepare for Keras voxnet model.
    Formats using EventData() class from EventData.py.
    """

    Data = EventData()
    Data.ID = SimEvent.GetID()
    
    #Set type, detector
    if SimEvent.GetNIAs() > 0: #what is this if statement doing?
        if SimEvent.GetIAAt(1).GetProcess() == M.MString("COMP"):
            Data.type = 0
        if SimEvent.GetIAAt(1).GetProcess() == M.MString("PAIR"):
            Data.type = 1

        Data.detector = SimEvent.GetIAAt(1).GetDetectorType()
        Data.GammaEnergy = SimEvent.GetIAAt(0).GetSecondaryEnergy()
  
    # Calculate total measured energy, create list of "hits" and x/y/z positions
    
    total_measured_energy = 0
    Hits = np.zeros((SimEvent.GetNHTs(), 4))
    for i in range(0, SimEvent.GetNHTs()):
        x_pos = SimEvent.GetHTAt(i).GetPosition().X()
        y_pos = SimEvent.GetHTAt(i).GetPosition().Y()
        z_pos = SimEvent.GetHTAt(i).GetPosition().Z()
        Hits[i, 0] = x_pos
        Hits[i, 1] = y_pos
        Hits[i, 2] = x_pos
        #Data.startX = np.append(Data.startX, np.array([x_pos]))
        #Data.startY = np.append(Data.startY, np.array([y_pos]))
        #Data.startZ = np.append(Data.startZ, np.array([z_pos]))
        hitEnergy = SimEvent.GetHTAt(i).GetEnergy()
        Hits[i, 3] = hitEnergy
        #save hits in eventdata, not startx/starty/startz
        total_measured_energy += hitEnergy

    Data.Hits = Hits
    Data.measured_energy = total_measured_energy

    return Data

# Select parser to use based upon arguments:

parser_options = {'voxnet':parse_voxnet}
parser_to_use = parser_options[args.parser]

# Load geometry:

# Geometry to use. Fixed for the time being
GeometryName = "$(MEGALIB)/resource/examples/geomega/GRIPS/GRIPS_extended.geo.setup"

Geometry = M.MDGeometryQuest()
if Geometry.ScanSetupFile(M.MString(GeometryName)) == True:
    print("Geometry " + GeometryName + " loaded!")
else:
    print("Unable to load geometry " + GeometryName + " - Aborting!")
    quit() #switch away from quit()?

Reader = M.MFileEventsSim(Geometry)
if Reader.Open(M.MString(self.FileName)) == False:
    print("Unable to open file " + FileName + ". Aborting!")
    quit() #switch away from quit()?

# Handle Ctrl-C:

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

#Loop for actual data processing:

DataSets = []

NumberOfEvents = 0
NumberOfDataSets = 0

print("\n\nStarted reading data sets")
while True:
    Event = Reader.GetNextEvent()
  
    if not Event:
        break

    NumberOfEvents += 1
    Data = parser_to_use(Event, Debug)

    if Data is not None:
        DataSets.append(Data)
        NumberOfDataSets += 1

    if NumberOfDataSets > 0 and NumberOfEvents % 1000 == 0:
        print("Data sets processed: {} / {}".format(NumberOfDataSets, NumberOfEvents))

    if NumberOfDataSets >= MaxDataSets:
        break

    if Interrupted == True:
        break

print("Info: Parsed {} events".format(NumberOfDataSets))

#Store data
print("Info: Storing the data")
with open(OutputFileName, "wb") as FileHandle:
    pickle.dump(DataSets, FileHandle)
print("Info: Done")

#input("Press [enter] to EXIT")
sys.exit(0)
