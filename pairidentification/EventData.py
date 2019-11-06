###################################################################################################
#
# Event Data
#
# Copyright (C) by Andreas Zoglauer.
#
# Please see the file License.txt in the main repository for the copyright-notice.
#
###################################################################################################




###################################################################################################


import random
import math
import numpy as np
import ROOT as M
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")


###################################################################################################


class EventData:
  """
  This class stores the data of one event
  """


###################################################################################################


  def __init__(self):
    """
    The default constructor for class EventData
    """

    self.ID = 0

    self.GammaEnergy = 0
  
    self.OriginPositionZ = 0.0
    
    self.X = np.zeros(shape=(0), dtype=float)
    self.Y = np.zeros(shape=(0), dtype=float)
    self.Z = np.zeros(shape=(0), dtype=float)
    self.E = np.zeros(shape=(0), dtype=float)



###################################################################################################


  def parse(self, SimEvent):
    """
    Extract the data from the MSimEvent class
    """

    self.ID = SimEvent.GetID()

    if SimEvent.GetNIAs() > 2 and SimEvent.GetNHTs() > 2:
      
      '''
      OnlyOneLayer = True
      zFirst = -1000 
      for i in range(0, SimEvent.GetNHTs()):
        if SimEvent.GetHTAt(i).GetDetectorType() == 1:
          if zFirst == -1000:
            zFirst = SimEvent.GetHTAt(i).GetPosition().Z()
            continue
          if math.fabs(zFirst - SimEvent.GetHTAt(i).GetPosition().Z()) > 0.01:
            OnlyOneLayer = False
            break
      '''
      
      self.GammaEnergy = SimEvent.GetIAAt(0).GetSecondaryEnergy()

      if SimEvent.GetIAAt(1).GetProcess() == M.MString("PAIR") and SimEvent.GetIAAt(1).GetDetectorType() == 1:
        
        Counter = 0
        for i in range(0, SimEvent.GetNHTs()):
          if SimEvent.GetHTAt(i).GetDetectorType() == 1 and SimEvent.GetHTAt(i).IsOrigin(2) == True:
            Counter += 1
        
        if Counter == 0:
          return False
        
        self.X = np.zeros(shape=(Counter), dtype=float)
        self.Y = np.zeros(shape=(Counter), dtype=float)
        self.Z = np.zeros(shape=(Counter), dtype=float)
        self.E = np.zeros(shape=(Counter), dtype=float)
        
        self.OriginPositionZ = SimEvent.GetIAAt(1).GetPosition().Z()
        
        IsOriginIncluded = False
        
        ZMin = 1000
        ZMax = -1000
        
        Counter = 0
        for i in range(0, SimEvent.GetNHTs()):
          if SimEvent.GetHTAt(i).GetDetectorType() == 1 and SimEvent.GetHTAt(i).IsOrigin(2) == True:
            self.X[Counter] = SimEvent.GetHTAt(i).GetPosition().X()        
            self.Y[Counter] = SimEvent.GetHTAt(i).GetPosition().Y()        
            self.Z[Counter] = SimEvent.GetHTAt(i).GetPosition().Z()        
            self.E[Counter] = SimEvent.GetHTAt(i).GetEnergy()
            
            if self.Z[Counter] < ZMin:
              ZMin = self.Z[Counter]
            
            if self.Z[Counter] > ZMax:
              ZMax = self.Z[Counter]
            
            if math.fabs(self.Z[Counter] - self.OriginPositionZ) < 0.1:
              IsOriginIncluded = True
            
            Counter += 1
          
        if IsOriginIncluded == False:
          return False
        
        # Pick out just 2-site events
        # ZDistance = ZMax - ZMin
        # NSites=5
        # if ZDistance > (NSites-0.5)*0.5 or ZDistance < (NSites-1.5)*0.5:
        #  return False
        
        
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
    """
    Returns True if any event are ouside the box defined by x in [XMin,XMax], y in [YMin,YMax]
    """
    
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
    """
    Print the data
    """

    print("Event ID: {}".format(self.ID))
    print("  Origin Z: {}".format(self.OriginPositionZ))
    for h in range(0, len(self.X)):
      print("  Hit {}: ({}, {}, {}), {} keV".format(h, self.X[h], self.Y[h], self.Z[h], self.E[h]))
      
      
      
      
      
      
      
      



