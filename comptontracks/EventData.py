###################################################################################################
#
# GRBData.py
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
    The default constructor for class EventClustering

    Attributes
    ----------
    FileName : string
      Data file name (something like: X.maxhits2.eventclusterizer.root)
    OutputPrefix: string
      Output filename prefix as well as outout directory name
    Algorithms: string
      The algorithms used during training. Seperate multiples by commma (e.g. "MLP,DNNCPU")
    MaxEvents: integer
      The maximum amount of events to use

    """

    self.OriginPosition = 0.0 
    
    self.X = np.zeros(shape=(0), dtype=float)
    self.Y = np.zeros(shape=(0), dtype=float)
    self.Z = np.zeros(shape=(0), dtype=float)
    self.E = np.zeros(shape=(0), dtype=float)



###################################################################################################


  def parse(self, SimEvent):
    """
    Switch between the various machine-learning libraries based on self.Algorithm
    """

    if SimEvent.GetNIAs() > 2 and SimEvent.GetNHTs() > 2:
      if SimEvent.GetIAAt(1).GetProcess() == M.MString("COMP") and SimEvent.GetIAAt(1).GetDetectorType() == 1:
        self.X = np.zeros(shape=(SimEvent.GetNHTs()), dtype=float)
        self.Y = np.zeros(shape=(SimEvent.GetNHTs()), dtype=float)
        self.Z = np.zeros(shape=(SimEvent.GetNHTs()), dtype=float)
        self.E = np.zeros(shape=(SimEvent.GetNHTs()), dtype=float)
        
        for i in range(0, SimEvent.GetNHTs()):
          self.X[i] = SimEvent.GetHTAt(0).GetPosition().X()        
          self.Y[i] = SimEvent.GetHTAt(0).GetPosition().Y()        
          self.Z[i] = SimEvent.GetHTAt(0).GetPosition().Z()        
          self.E[i] = SimEvent.GetHTAt(0).GetEnergy()
          
        self.OriginPosition = SimEvent.GetIAAt(1).GetPosition().Z()
      else:
        return False
    else:
      return False
    
    return True
      
      
      
      
      
      
      
      



