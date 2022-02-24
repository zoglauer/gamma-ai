import ROOT
import array
import os
import sys
import random
import time
import collections
import numpy as np
import math, datetime
from tqdm import tqdm
import pickle
from voxnet import *
from volumetric_data import ShapeNet40Vox30
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


#Step 1: Input parameters

XBins = 110
YBins = 110
ZBins = 48
BatchSize = 20
XMin = -55
XMax = 55
YMin = -55
YMax = 55
ZMin = 0
ZMax = 48
outputDirectory = "output.txt"
train_test_split = 0.9
MaxLabel = 0


#Step 2: Global functions
# really confused about the way data was being loaded in so this is probably wrong sorry :(

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
        return EventEnergies[:self.MaxEvents], GammaEnergies[:self.MaxEvents]

    if not self.DataLoaded:
      self.loadData()
    return self.EventEnergies, self.GammaEnergies



#Step 3: Split batches into training and testing

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

#Step 4: initialize neural network
# conv + leakyrelu (speed training) + maxpooling + normalize + flatten + dense

Model = models.Sequential()
Model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(XBins, YBins, ZBins, 1)))
Model.add(layers.LeakyReLU(alpha = 0.1))
Model.add(layers.MaxPooling3D((2, 2, 3)))
Model.add(layers.LayerNormalization(axis=1))
Model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
Model.add(layers.LeakyReLU(alpha = 0.1))
Model.add(layers.MaxPooling3D((2, 2, 2)))
Model.add(layers.LayerNormalization(axis=1))
Model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))

Model.add(layers.Flatten())
Model.add(layers.Dense(64, activation='relu'))
Model.add(layers.Dense(64, activation = 'relu'))
#Model.add(layers.Dense(64, activation = 'Softmax'))

Model.compile(optimizer = 'adam', loss = '', metrics = ['accuracy'])

Model.summary()

#Step 5: train network

    TimeConverting = 0.0
    TimeTraining = 0.0
    TimeTesting = 0.0
    Iteration = 0
    MaxIterations = 50000
    TimesNoImprovement = 0
    MaxTimesNoImprovement = 50
    while (Iteration < MaxIterations):
        Iteration += 1
        print("\n\nStarting iteration {}".format(Iteration))
        for Batch in range(0, NTrainingBatches):
          print("Batch {} / {}".format(Batch+1, NTrainingBatches))
          TimerConverting = time.time()

          for g in range(0, BatchSize):
              Event = TrainingDataSets[g + Batch*BatchSize]
              if Event.OriginPositionZ > self.ZMin and Event.OriginPositionZ < self.ZMax:
                    LayerBin = int ((Event.OriginPositionZ - self.ZMin) / ((self.ZMax- self.ZMin)/ self.ZBins) )
                    OutputTensor[g][LayerBin] = 1
              else:
                OutputTensor[g][self.dataLoader.OutputDataSpaceSize-1] = 1
          for h in range(0, len(Event.X)):
              XBin = int( (Event.X[h] - XMin) / ((XMax - XMin) / XBins) )
              YBin = int( (Event.Y[h] - YMin) / ((YMax - YMin) / YBins) )
              ZBin = int( (Event.Z[h] - ZMin) / ((ZMax - ZMin) / ZBins) )
              if XBin >= 0 and YBin >= 0 and ZBin >= 0 and XBin < XBins and YBin < YBins and ZBin < ZBins:
                  InputTensor[g][XBin][YBin][ZBin][0] = Event.E[h]

          TimeConverting += time.time() - TimerConverting

          TimerTraining = time.time()
          InputTensor = np.zeros(shape=(BatchSize, XBins, YBins, ZBins, 1))
          OutputTensor = np.zeros(shape=(BatchSize, 64))
          trained = Model.fit(InputTensor,OutputTensor, epochs=10, validation_split=0.1)
          loss = trained.evaluate()
          TimeTraining += time.time() - TimerTraining


        #check performance

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

          # Take care of Ctrl-C
          if Interrupted == True: break

# Step 6: test network
#havent fixed yet
  for Batch in range(0, NTestingBatches):

    # Step 1.1: Convert the data set into the input and output tensor
    InputTensor = np.zeros(shape=(BatchSize, XBins, YBins, ZBins, 1))
    OutputTensor = np.zeros(shape=(BatchSize, OutputDataSpaceSize))


    # Loop over all testing  data sets and add them to the tensor
    for e in range(0, BatchSize):
      Event = TestingDataSets[e + Batch*BatchSize]
      # Set the layer in which the event happened
      if Event.OriginPositionZ > self.ZMin and Event.OriginPositionZ < self.ZMax:
        LayerBin = int ((Event.OriginPositionZ - self.ZMin) / ((self.ZMax- self.ZMin)/ self.ZBins) )
        #print("layer bin: {} {}".format(Event.OriginPositionZ, LayerBin))
        OutputTensor[e][LayerBin] = 1
      else:
        OutputTensor[e][self.dataLoader.OutputDataSpaceSize-1] = 1

      # Set all the hit locations and energies
      SomethingAdded = False
      for h in range(0, len(Event.X)):
        XBin = int( (Event.X[h] - self.XMin) / ((self.XMax - self.XMin) / self.XBins) )
        YBin = int( (Event.Y[h] - self.YMin) / ((self.YMax - self.YMin) / self.YBins) )
        ZBin = int( (Event.Z[h] - self.ZMin) / ((self.ZMax - self.ZMin) / self.ZBins) )
        #print("hit z bin: {} {}".format(Event.Z[h], ZBin))
        if XBin >= 0 and YBin >= 0 and ZBin >= 0 and XBin < self.XBins and YBin < self.YBins and ZBin < self.ZBins:
          InputTensor[e][XBin][YBin][ZBin][0] = Event.E[h]
          SomethingAdded = True

      if SomethingAdded == False:
        print("Nothing added for event {}".format(Event.ID))
        Event.print()


    # Step 2: Run it
    # Result = Session.run(Output, feed_dict={X: InputTensor})
    Result = Model.predict(InputTensor)

    #print(Result[e])
    #print(OutputTensor[e])

    for e in range(0, BatchSize):
      TotalEvents += 1
      IsBad = False
      LargestValueBin = 0
      LargestValue = OutputTensor[e][0]
      for c in range(1, self.dataLoader.OutputDataSpaceSize) :
        if Result[e][c] > LargestValue:
          LargestValue = Result[e][c]
          LargestValueBin = c

      if OutputTensor[e][LargestValueBin] < 0.99:
        BadEvents += 1
        IsBad = True

        #if math.fabs(Result[e][c] - OutputTensor[e][c]) > 0.1:
        #  BadEvents += 1
        #  IsBad = True
        #  break

      # Fetch real and predicted layers for testing data
)

      # Some debugging
      if Batch == 0 and e < 500:
        EventID = e + Batch*BatchSize + NTrainingBatches*BatchSize
        print("Event {}:".format(EventID))
        if IsBad == True:
          print("BAD")
        else:
          print("GOOD")
        DataSets[EventID].print()

        print("Results layer: {}".format(LargestValueBin))
        for l in range(0, self.dataLoader.OutputDataSpaceSize):
          if OutputTensor[e][l] > 0.5:
            print("Real layer: {}".format(l))
          #print(OutputTensor[e])
          #print(Result[e])

  PercentageGood = 100.0 * float(TotalEvents-BadEvents) / TotalEvents

  if PercentageGood > BestPercentageGood:
    BestPercentageGood = PercentageGood
    Improvement = True

  print("Percentage of good events: {:-6.2f}% (best so far: {:-6.2f}%)".format(PercentageGood, BestPercentageGood))

  return Improvement
