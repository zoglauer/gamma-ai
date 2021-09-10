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

    self.TrackStartX = 0.0
    self.TrackStartY = 0.0
    self.TrackStartZ = 0.0

    self.TrackDirectionX = 0.0
    self.TrackDirectionY = 0.0
    self.TrackDirectionZ = 0.0

    self.X      = np.zeros(shape=(self.MaxHits), dtype=float)
    self.Y      = np.zeros(shape=(self.MaxHits), dtype=float)
    self.Z      = np.zeros(shape=(self.MaxHits), dtype=float)
    self.E      = np.zeros(shape=(self.MaxHits), dtype=float)



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

    #print("{}: [{}, {}], {}: [{}, {}], {}: [{}, {}]".format(self.TrackStartX, XMin, XMax, self.TrackStartY, YMin, YMax, self.TrackStartZ, ZMin, ZMax))

    if self.TrackStartX > XMax:
      return False
    if self.TrackStartX < XMin:
      return False
    if self.TrackStartY > YMax:
      return False
    if self.TrackStartY < YMin:
      return False
    if self.TrackStartZ > XMax:
      return False
    if self.TrackStartZ < ZMin:
      return False

    return True


###################################################################################################


  def print(self):
    """
    Print the data
    """

    print("Event ID: {}".format(self.ID))
    print("  Start: {} {} {}".format(self.TrackStartX, self.TrackStartY, self.TrackStartZ))
    print("  Dir:   {} {} {}".format(self.TrackDirectionX, self.TrackDirectionY, self.TrackDirectionZ))
    for h in range(0, len(self.X)):
      print("  Hit {}: pos=({}, {}, {})cm, E={}keV".format(h, self.X[h], self.Y[h], self.Z[h], self.E[h]))
      


###################################################################################################

      
