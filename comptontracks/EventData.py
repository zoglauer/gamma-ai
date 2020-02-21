###################################################################################################
#
# EventData
#
# Copyright (C) by Andreas Zoglauer.
# All rights reserved.
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

    self.MaxHits = 100

    self.EventID = 0
    self.unique = 0

    self.OriginPositionX = 0.0
    self.OriginPositionY = 0.0
    self.OriginPositionZ = 0.0

    self.ID     = np.zeros(shape=(self.MaxHits), dtype=int)
    self.Origin = np.zeros(shape=(self.MaxHits), dtype=int)
    self.X      = np.zeros(shape=(self.MaxHits), dtype=float)
    self.Y      = np.zeros(shape=(self.MaxHits), dtype=float)
    self.Z      = np.zeros(shape=(self.MaxHits), dtype=float)
    self.E      = np.zeros(shape=(self.MaxHits), dtype=float)
    self.Type   = np.zeros(shape=(self.MaxHits), dtype=str)


###################################################################################################


  def createFromToyModel(self, EventID):

    self.EventID = EventID

    # Step 1: Simulate the gamma ray according to Butcher & Messel: Nuc Phys 20(1960), 15
    
    # Initial energy
    Ei = 2000 

    # Random initial direction
    Di = M.MVector()
    Di.SetMagThetaPhi(1.0, np.arccos(1 - 2*random.random()), 2.0 * np.pi * random.random())

    # Start position (randomly within a certian volume)
    xi = 40.0 * (random.random() - 0.5)
    yi = 40.0 * (random.random() - 0.5)
    zi = int(40.0 * (random.random() - 0.5))

    print("Start: {}, {}, {}".format(xi, yi, zi))

    self.OriginPositionX = xi
    self.OriginPositionY = yi
    self.OriginPositionZ = zi
 
    E0 = 510.998910
    Ei_m = Ei / E0

    Epsilon = 0.0
    EpsilonSquare = 0.0
    OneMinusCosTheta = 0.0
    SinThetaSquared = 0.0

    Epsilon0 = 1./(1. + 2.*Ei_m)
    Epsilon0Square = Epsilon0*Epsilon0
    Alpha1 = - math.log(Epsilon0)
    Alpha2 = 0.5*(1.- Epsilon0Square)

    Reject = 0.0

    while True:
      if Alpha1/(Alpha1+Alpha2) > random.random():
        Epsilon = math.exp(-Alpha1*random.random())
        EpsilonSquare = Epsilon*Epsilon
      else:
        EpsilonSquare = Epsilon0Square + (1.0 - Epsilon0Square)*random.random()
        Epsilon = math.sqrt(EpsilonSquare)
      
      OneMinusCosTheta = (1.- Epsilon)/(Epsilon*Ei_m)
      SinThetaSquared = OneMinusCosTheta*(2.-OneMinusCosTheta)
      Reject = 1.0 - Epsilon*SinThetaSquared/(1.0 + EpsilonSquare)

      if Reject < random.random():
        break
  
    CosTheta = 1.0 - OneMinusCosTheta; 
    SinTeta = math.sqrt(SinThetaSquared);
    Phi = 2*math.pi * random.random();
  

    # Set the new photon and electron parameters relative to original direction
    Eg = Epsilon*Ei
    Ee = Ei - Eg

    Dg = M.MVector(SinTeta*math.cos(Phi), SinTeta*math.sin(Phi), CosTheta);
    Dg.RotateReferenceFrame(Di);


    Me = math.sqrt(Ee*(Ee+2.0*E0));
    De = (Ei * Di - Eg * Dg) * (1.0 / Me);


    # Track the electron
    xe = xi
    ye = yi
    ze = zi
    IsInitial = 0
    ID = 1
    while Ee > 0 and ID < self.MaxHits - 2:
      dE = 0
      while dE <= 0:
        dE = random.gauss(10*math.sqrt(Ei-Ee), 0.1*math.sqrt(Ee))
        
      if ID == 1:
        dE *= random.random()
        
      if dE > Ee:
        dE = Ee
      
      
      #print("electron track {} with {} {} {} {} & {}".format(ID, xe, ye, ze, Ee, dE))

      self.Origin[ID-1] = ID - 1
      self.ID[ID-1] = ID
      self.X[ID-1] = xe
      self.Y[ID-1] = ye
      self.Z[ID-1] = ze
      self.E[ID-1] = dE
      if ID == 1:
        self.Type[ID-1] = "eg"
      else: 
        self.Type[ID-1] = "e"
      
      ID += 1
      Ee -= dE
    
      dAngle = (Ei - Ee) * 0.4*math.pi / Ei
      
      dEe = M.MVector()
      dEe.SetMagThetaPhi(1.0, dAngle, 2.0 * np.pi * random.random())
      
      De.RotateReferenceFrame(dEe);
      
      Distance = 2.0 + 3.0 * random.random()
      
      xe += Distance * De.X()
      ye += Distance * De.Y()
      ze += Distance * De.Z()
      
    
    # Track the gamma ray
    Origin = 1
    
    Distance = 10.0 + 10.0 * random.random()
  
    self.Origin[ID-1] = 1   
    self.ID[ID-1] = ID
    self.X[ID-1] = xi + Distance * Dg.X()
    self.Y[ID-1] = yi + Distance * Dg.Y()
    self.Z[ID-1] = zi + Distance * Dg.Z()
    self.E[ID-1] = Eg
    self.Type[ID-1] = "g"
  
  
    # Shrink
    self.Origin.resize(ID)
    self.ID.resize(ID)
    self.X.resize(ID)
    self.Y.resize(ID)
    self.Z.resize(ID)
    self.E.resize(ID)
    self.Type.resize(ID)
  
    self.print()
  
    return

    

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

      if SimEvent.GetIAAt(1).GetProcess() == M.MString("COMP") and SimEvent.GetIAAt(1).GetDetectorType() == 1 and SimEvent.GetNGRs() == 0 and SimEvent.IsIACompletelyAbsorbed(1, 10.0, 2.0):

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

        self.OriginPositionX = SimEvent.GetIAAt(1).GetPosition().X()
        self.OriginPositionY = SimEvent.GetIAAt(1).GetPosition().Y()
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
        #   return False

        self.unique = len(np.unique(self.Z))
        # if (self.unique == 1): return False

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

    #print("{}: [{}, {}], {}: [{}, {}], {}: [{}, {}]".format(self.OriginPositionX, XMin, XMax, self.OriginPositionY, YMin, YMax, self.OriginPositionZ, ZMin, ZMax))

    if self.OriginPositionX > XMax:
      return False
    if self.OriginPositionX < XMin:
      return False
    if self.OriginPositionY > YMax:
      return False
    if self.OriginPositionY < YMin:
      return False
    if self.OriginPositionZ > XMax:
      return False
    if self.OriginPositionZ < ZMin:
      return False

    return True


###################################################################################################


  def print(self):
    """
    Print the data
    """

    print("Event ID: {}".format(self.EventID))
    print("  Origin Z: {}".format(self.OriginPositionZ))
    for h in range(0, len(self.X)):
      print("  Hit {} (origin: {}): type={}, pos=({}, {}, {})cm, E={}keV".format(self.ID[h], self.Origin[h], self.Type[h], self.X[h], self.Y[h], self.Z[h], self.E[h]))
