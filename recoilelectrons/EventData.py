###################################################################################################
#
# EventData
#
# Copyright (C) by Andreas Zoglauer & contributors
# All rights reserved.
#
# Please see the file License.txt in the main repository for the copyright-notice.
#
###################################################################################################




###################################################################################################


import numpy as np


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
    
    self.MaxHits = 100
    
    self.ID = 0

    self.TrackRealStartX = 0.0
    self.TrackRealStartY = 0.0
    self.TrackRealStartZ = 0.0

    self.TrackRealDirectionX = 0.0
    self.TrackRealDirectionY = 0.0
    self.TrackRealDirectionZ = 0.0

    self.TrackMeasuredStartX = 0.0
    self.TrackMeasuredStartY = 0.0
    self.TrackMeasuredStartZ = 0.0

    self.TrackMeasuredDirectionX = 0.0
    self.TrackMeasuredDirectionY = 0.0
    self.TrackMeasuredDirectionZ = 0.0

    self.TrackSequence = np.zeros(shape=(self.MaxHits), dtype=int)

    self.X = np.zeros(shape=(self.MaxHits), dtype=float)
    self.Y = np.zeros(shape=(self.MaxHits), dtype=float)
    self.Z = np.zeros(shape=(self.MaxHits), dtype=float)
    self.E = np.zeros(shape=(self.MaxHits), dtype=float)
    



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


  def hasHitsOutside(self, XMin, XMax, YMin, YMax, ZMin, ZMax):
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

    for e in range(0, len(self.Z)):
      if self.Z[e] > ZMax:
        return True
      if self.Z[e] < ZMin:
        return True

    return False


###################################################################################################


  def isOriginInside(self, XMin, XMax, YMin, YMax, ZMin, ZMax):
    """
    Returns True if the start is inside the box defined by x in [XMin,XMax], y in [YMin,YMax], z in [ZMin,ZMax]
    """

    #print("{}: [{}, {}], {}: [{}, {}], {}: [{}, {}]".format(self.TrackRealStartX, XMin, XMax, self.TrackRealStartY, YMin, YMax, self.TrackRealStartZ, ZMin, ZMax))

    if self.TrackRealStartX > XMax:
      return False
    if self.TrackRealStartX < XMin:
      return False
    if self.TrackRealStartY > YMax:
      return False
    if self.TrackRealStartY < YMin:
      return False
    if self.TrackRealStartZ > XMax:
      return False
    if self.TrackRealStartZ < ZMin:
      return False

    return True


###################################################################################################


  def print(self):
    """
    Print the data
    """

    print("Event ID: {}".format(self.ID))
    print("  Real Start:         {:+.4f} {:+.4f} {:+.4f}".format(self.TrackRealStartX, self.TrackRealStartY, self.TrackRealStartZ))
    print("  Measured Start:     {:+.4f} {:+.4f} {:+.4f}".format(self.TrackMeasuredStartX, self.TrackMeasuredStartY, self.TrackMeasuredStartZ))
    print("  Real direction:     {:+.4f} {:+.4f} {:+.4f}".format(self.TrackRealDirectionX, self.TrackRealDirectionY, self.TrackRealDirectionZ))
    print("  Measured direction: {:+.4f} {:+.4f} {:+.4f}".format(self.TrackMeasuredDirectionX, self.TrackMeasuredDirectionY, self.TrackMeasuredDirectionZ))
    for h in range(0, len(self.X)):
      print("  Hit {}: pos=({:+.4f}, {:+.4f}, {:+.4f})cm, E={:5.2f}keV, seq={}".format(h, self.X[h], self.Y[h], self.Z[h], self.E[h], self.TrackSequence[h]))
      


###################################################################################################

      
