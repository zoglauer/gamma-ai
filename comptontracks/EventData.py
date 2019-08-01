###################################################################################################
#
# GRBpy
#
# Copyright (C) by Andreas Zoglauer.
# All rights reserved.
#
# Please see the file License.txt in the main repository for the copyright-notice.
#
###################################################################################################




###################################################################################################


import random 
import numpy as np
import ROOT as M
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")


###################################################################################################


class EventData:
  """
  This class performs energy loss training. A typical usage would look like this:

  AI = EventTypeIdentification("Ling2.seq3.quality.root", "Results", "TF:VOXNET", 1000000)
  AI.train()
  AI.test()

  """


###################################################################################################


  def __init__(self):
    """
    The default constructor for class EventData
    """

    self.ID = 0

    self.OriginPositionZ = 0.0
    
    self.X = np.zeros(shape=(0), dtype=float)
    self.Y = np.zeros(shape=(0), dtype=float)
    self.Z = np.zeros(shape=(0), dtype=float)
    self.E = np.zeros(shape=(0), dtype=float)



###################################################################################################


  def parse(self, SimEvent):
    """
    Switch between the various machine-learning libraries based on self.Algorithm
    """

    self.ID = SimEvent.GetID()

    if SimEvent.GetNIAs() > 2 and SimEvent.GetNHTs() > 2:
      if SimEvent.GetIAAt(1).GetProcess() == M.MString("COMP") and SimEvent.GetIAAt(1).GetDetectorType() == 1:
        
        Counter = 0
        for i in range(0, SimEvent.GetNHTs()):
          if SimEvent.GetHTAt(i).GetDetectorType() == 1 and SimEvent.GetHTAt(i).IsOrigin(2) == True and SimEvent.GetNGRs() == 0:
            Counter += 1
        
        if Counter == 0:
          return False
        
        self.X = np.zeros(shape=(Counter), dtype=float)
        self.Y = np.zeros(shape=(Counter), dtype=float)
        self.Z = np.zeros(shape=(Counter), dtype=float)
        self.E = np.zeros(shape=(Counter), dtype=float)
        
        Counter = 0
        for i in range(0, SimEvent.GetNHTs()):
          if SimEvent.GetHTAt(i).GetDetectorType() == 1 and SimEvent.GetHTAt(i).IsOrigin(2) == True and SimEvent.GetNGRs() == 0:
            self.X[Counter] = SimEvent.GetHTAt(i).GetPosition().X()        
            self.Y[Counter] = SimEvent.GetHTAt(i).GetPosition().Y()        
            self.Z[Counter] = SimEvent.GetHTAt(i).GetPosition().Z()        
            self.E[Counter] = SimEvent.GetHTAt(i).GetEnergy()
            Counter += 1
          
        self.OriginPositionZ = SimEvent.GetIAAt(1).GetPosition().Z()
      else:
        return False
    else:
      return False
    
    return True



###################################################################################################


  def center(self):
    """
    Move the center of the track to 0/0
    """
    
    XExtentMin = 1000
    XExtentMax = -1000
    for e in range(0, len(self.X)):
      if self.X[e] > XExtentMax: 
        XExtentMax = self.X[e]
      if self.X[e] < XExtentMin:
        XExtentMin = self.X[e]
      
    XCenter = 0.5*(XExtentMin + XExtentMax)
  
    YExtentMin = 1000
    YExtentMax = -1000
    for e in range(0, len(self.Y)):
      if self.Y[e] > YExtentMax: 
        YExtentMax = self.Y[e]
      if self.Y[e] < YExtentMin:
        YExtentMin = self.Y[e]
      
    YCenter = 0.5*(YExtentMin + YExtentMax)
  
    for e in range(0, len(self.X)):
      self.X[e] -= XCenter
  
    for e in range(0, len(self.Y)):
      self.Y[e] -= YCenter


###################################################################################################


  def hasHitsOutside(self, XMin, XMax, YMin, YMax):

    for e in range(0, len(self.X)):
      if self.X[e] > XMax: 
        return True
      if self.X[e] < XMin:
        return True

    for e in range(0, len(self.Y)):
      if self.Y[e] > YMax: 
        return True
      if self.Y[e] < YMin:
        return True

    return False
  

###################################################################################################


  def print(self):
    print("Event ID: {}".format(self.ID))
    print("  Origin Z: {}".format(self.OriginPositionZ))
    for h in range(0, len(self.X)):
      print("  Hit {}: ({}, {}, {}), {} keV".format(h, self.X[h], self.Y[h], self.Z[h], self.E[h]))
      
      
      
      
      
      
      
      



