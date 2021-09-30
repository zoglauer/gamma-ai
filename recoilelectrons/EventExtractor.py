###################################################################################################
#
# EventExtractor.py
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
import os
import argparse


import numpy as np

# Everything ROOT related can only be loaded here otherwise it interferes with the argparse
from EventData import EventData

# Load MEGAlib into ROOT so that it is usable
import ROOT as M
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")
M.PyConfig.IgnoreCommandLineOptions = True



print("\nEvent Extractor")
print("============================\n")



# Step 1: Input parameters
###################################################################################################


# Default parameters

GeometryName = "$(MEGALIB)/resource/examples/geomega/GRIPS/GRIPS.geo.setup"

parser = argparse.ArgumentParser(description='Extract events from Cosima sim file.')
parser.add_argument('-f', '--filename', default='RecoilElectrons.p1.sim.gz', help='File name used for training/testing')
parser.add_argument('-m', '--maxdatasets', default=1000000000, type=int, help='Maximum number of good data sets to extract')
parser.add_argument('-d', '--debug', default=False, action="store_true", help='Enable debugging output')

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
# Step 3: Read the data
###################################################################################################



###################################################################################################


def parse(SimEvent, Debug):
    """
    Extract the data from the MSimEvent class
    """

    Data = EventData()

    Data.ID = SimEvent.GetID()


    # Clusterize adjacent strip hits
    SimEvent.CreateClusters()
    Hits = [] 
    HitSequence = []
    for c in range(0, SimEvent.GetNClusters()):
      HT = SimEvent.GetClusterAt(c).CreateHT()
      M.SetOwnership(HT, True) # Python needs ownership of the event in order to delete it
      Hits.append(HT)

    for c in range(0, SimEvent.GetNClusters()):
      HitSequence.append(1000000)
      for h in range(0, SimEvent.GetNHTs()):
        if SimEvent.GetClusterAt(c).HasHT(SimEvent.GetHTAt(h)) == True: 
          if h < HitSequence[c]:
            HitSequence[c] = h
      
    SimEvent.DeleteAllHTs()

    for h in range(0, len(Hits)):
      SimEvent.AddHT(Hits[h])

    # Only pick good events
    if SimEvent.GetNIAs() <= 3:
      if Debug == True: print("Event {} rejected: Not enough IAs: {}".format(self.ID, SimEvent.GetNIAs()))
      return None

    if SimEvent.GetNHTs() < 3:
      if Debug == True: print("Event {} rejected: Not enough hits: {}".format(self.ID, SimEvent.GetNHTs()))
      return None

    if SimEvent.GetIAAt(1).GetProcess() != M.MString("COMP"):
      if Debug == True: print("Event {} rejected: First interaction not Compton: {}".format(self.ID, SimEvent.GetIAAt(1).GetProcess().Data()))
      return None

    if SimEvent.GetIAAt(1).GetDetectorType() != 1 and SimEvent.GetIAAt(1).GetDetectorType() != 3:
      if Debug == True: print("Event {} rejected: First interaction not in strip detector: {}".format(self.ID, SimEvent.GetIAAt(1).GetDetectorType()))
      return None

    if SimEvent.GetIAAt(2).GetDetectorType() == 1:
      if Debug == True: print("Event {} rejected: Second interaction in tracker".format(self.ID))
      return None

    if SimEvent.GetNPMs() > 0:
      if Debug == True: print("Event {} rejected: Energy deposits in passive material found".format(self.ID))
      return None

    if SimEvent.IsIACompletelyAbsorbed(1, 10.0, 2.0) == False:
      if Debug == True: print("Event {} rejected: Not completely absorbed".format(self.ID))
      return None

    if SimEvent.GetNGRs() > 0:
      if Debug == True: print("Event {} rejected: Guard ring vetoes".format(self.ID))
      return None

    for i in range(0, SimEvent.GetNIAs()):
      if SimEvent.GetIAAt(i).GetProcess() == M.MString("PAIR"):
        if Debug == True: print("Event {} rejected: Pair interaction found".format(self.ID))
        return None
      if SimEvent.GetIAAt(i).GetProcess() == M.MString("BREM"):
        if Debug == True: print("Event {} rejected: Bremsstrahlung found".format(self.ID))
        return None
      if SimEvent.GetIAAt(i).GetProcess() == M.MString("RAYL"):
        if Debug == True: print("Event {} rejected: Rayleigh interaction found".format(self.ID))
        return None
      if SimEvent.GetIAAt(i).GetProcess() == M.MString("ESCP"):
        if Debug == True: print("Event {} rejected: Particle escape found".format(self.ID))
        return None

    # Create a list of all hits originating from the Compton track:
    IAIDs = SimEvent.GetDescendents(2);
    
    if IAIDs.size() < 2:
      if Debug == True: print("Event {} rejected: Track is not long enough".format(self.ID))
      return None

    TrackHits = []
    TrackHitSequence = []
    for h in range(0, len(Hits)):
      for i in IAIDs:
        if Hits[h].IsOrigin(i) == True:
          TrackHits.append(Hits[h])
          TrackHitSequence.append(HitSequence[h])
          break
        
    if len(TrackHits) < 2:
      if Debug == True: print("Event {} rejected: Track is not long enough".format(self.ID))
      return None

    Counter = len(TrackHits)

    # Is the start close to one of the hits
    MinDistance = 1000
    for i in range(0, Counter):
      Distance = (SimEvent.GetIAAt(1).GetPosition() - TrackHits[i].GetPosition()).Mag()
      if Distance < MinDistance:
        MinDistance = Distance
    if MinDistance > 0.1:
      if Debug == True: print("Event {} rejected: Track start is not in a hit".format(self.ID))
      return None


    # Resize the data
    Data.X = np.zeros(shape=(Counter), dtype=float)
    Data.Y = np.zeros(shape=(Counter), dtype=float)
    Data.Z = np.zeros(shape=(Counter), dtype=float)
    Data.E = np.zeros(shape=(Counter), dtype=float)
    Data.TrackSequence = np.zeros(shape=(Counter), dtype=float)

    # Save the data
    Data.TrackRealStartX = SimEvent.GetIAAt(1).GetPosition().X()
    Data.TrackRealStartY = SimEvent.GetIAAt(1).GetPosition().Y()
    Data.TrackRealStartZ = SimEvent.GetIAAt(1).GetPosition().Z()

    Data.TrackRealDirectionX = SimEvent.GetIAAt(1).GetSecondaryDirection().X()
    Data.TrackRealDirectionY = SimEvent.GetIAAt(1).GetSecondaryDirection().Y()
    Data.TrackRealDirectionZ = SimEvent.GetIAAt(1).GetSecondaryDirection().Z()

    for i in range(0, Counter):
      Data.X[i] = TrackHits[i].GetPosition().X()
      Data.Y[i] = TrackHits[i].GetPosition().Y()
      Data.Z[i] = TrackHits[i].GetPosition().Z()
      Data.E[i] = TrackHits[i].GetEnergy()
      
      Data.TrackSequence[i] = TrackHitSequence[i]
      
    Data.TrackMeasuredStartX = TrackHits[0].GetPosition().X()
    Data.TrackMeasuredStartY = TrackHits[0].GetPosition().Y()
    Data.TrackMeasuredStartZ = TrackHits[0].GetPosition().Z()

    MeasuredDir = M.MVector(TrackHits[1].GetPosition().X(), TrackHits[1].GetPosition().Y(), TrackHits[1].GetPosition().Z()) - M.MVector(TrackHits[0].GetPosition().X(), TrackHits[0].GetPosition().Y(), TrackHits[0].GetPosition().Z())
    MeasuredDir.Unitize()

    Data.TrackMeasuredDirectionX = MeasuredDir.X()
    Data.TrackMeasuredDirectionY = MeasuredDir.Y()
    Data.TrackMeasuredDirectionZ = MeasuredDir.Z()


    if Debug == True:
      print(SimEvent.ToSimString().Data())
      Data.print()

    return Data

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
NumberOfEvents = 0
NumberOfDataSets = 0
while True:
  Event = Reader.GetNextEvent()  
  if not Event:
    break
  M.SetOwnership(Event, True) # Python needs ownership of the event in order to delete it


  NumberOfEvents += 1
  
  Data = parse(Event, Debug)
  if Data is not None:
    #Data.print()
    DataSets.append(Data)
    NumberOfDataSets += 1

  if NumberOfDataSets > 0 and NumberOfEvents % 1000 == 0:
    print("Data sets processed: {} / {}".format(NumberOfDataSets, NumberOfEvents))

  if NumberOfDataSets >= MaxDataSets:
    break

  if Interrupted == True:
    break

print("Info: Parsed {} events".format(NumberOfDataSets))


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
