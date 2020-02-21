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

    self.MaxHits = 1000

    self.ID = 0

    self.GammaEnergy = 0

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
    Ei = 10000

    # Random initial direction
    Di = M.MVector()
    Di.SetMagThetaPhi(1.0, np.arccos(1 - 2*random.random()), 2.0 * np.pi * random.random())

    # Start position (randomly within a certain volume)
    xi = 40.0 * (random.random() - 0.5)
    yi = 40.0 * (random.random() - 0.5)
    zi = int(40.0 * (random.random() - 0.5))

    print("Start: {}, {}, {}".format(xi, yi, zi))

    self.OriginPositionX = xi
    self.OriginPositionY = yi
    self.OriginPositionZ = zi


    # Ranodm energy split
    Ee = (0.2 + random.random() * 0.8)*Ei 
    Ep = Ei - Ee
 
    # Random opening angle
    OpeningAngle = 0.1 + 0.6*random.random()
    
    # Initial diectrion electron and positron
    Pe = 2*math.pi * random.random()
    Te = math.pi - Ee/Ei * OpeningAngle
    De = M.MVector()
    De.SetMagThetaPhi(1.0, Te, Pe)

    Pp = Pe - math.pi
    Tp = math.pi - Ep/Ei * OpeningAngle
    Dp = M.MVector()
    Dp.SetMagThetaPhi(1.0, Tp, Pp)
 
 

    # Track the electron
    ID = 1
    InitialDepth = random.random()
    Origin = 0
    for t in [ "e", "p" ]:
      xe = xi
      ye = yi
      ze = zi
      IsInitial = True
      if t == "e":
        Energy = Ee
        Direction = De
      else:
        Energy = Ep
        Direction = Dp
      
      while Energy > 0 and ID < self.MaxHits - 2:
        dE = 0
        while dE <= 0:
          dE = max(random.gauss(250, 20), random.gauss(10*math.sqrt(Ei-Energy), 0.1*math.sqrt(Energy)))
        
        if IsInitial == True:
          dE *= InitialDepth
        
        if dE > Energy:
          dE = Energy
      
      
        #print("electron track {} with {} {} {} {} & {}".format(ID, xe, ye, ze, Energy, dE))
        
        if IsInitial:
          #print("ID: {} / {}, Edep: {}".format(1, 0, dE))
          self.Origin[0] = 0
          self.ID[0] = 1
          self.X[0] = xe
          self.Y[0] = ye
          self.Z[0] = ze
          self.E[0] += dE
          self.Type[0] = "m"
          IsInitial = False
          Origin = 1
          if t == "p":
            ID -= 1
            #print("eliminating ID for".format(t))
        else:
          #print("ID: {} / {}, Edep: {}".format(ID, ID-1, dE))
          self.Origin[ID-1] = Origin
          self.ID[ID-1] = ID
          self.X[ID-1] = xe
          self.Y[ID-1] = ye
          self.Z[ID-1] = ze
          self.E[ID-1] = dE
          self.Type[ID-1] = t
          Origin = ID
          
        
        ID += 1
        Energy -= dE
        self.GammaEnergy += dE
    
        dAngle = (Ei - Energy) * 0.4*math.pi / Ei
      
        dEe = M.MVector()
        dEe.SetMagThetaPhi(1.0, dAngle, 2.0 * np.pi * random.random())
      
        Direction.RotateReferenceFrame(dEe);
      
        if Direction.Z() > 0:
          ze_new = ze + 1
        else:
          ze_new = ze - 1
          
        Lambda = (ze_new - ze) / Direction.Z()
        
        xe += Lambda * Direction.X()
        ye += Lambda * Direction.Y()
        ze += Lambda * Direction.Z()
        
      

  
    # Shrink
    self.Origin.resize(ID-1)
    self.ID.resize(ID-1)
    self.X.resize(ID-1)
    self.Y.resize(ID-1)
    self.Z.resize(ID-1)
    self.E.resize(ID-1)
    self.Type.resize(ID-1)  
  
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

      self.GammaEnergy = SimEvent.GetIAAt(0).GetSecondaryEnergy()

      if SimEvent.GetIAAt(1).GetProcess() == M.MString("PAIR") and SimEvent.GetIAAt(1).GetDetectorType() == 1:

        Counter = 0
        for i in range(0, SimEvent.GetNHTs()):
          if SimEvent.GetHTAt(i).GetDetectorType() == 1 and SimEvent.GetHTAt(i).IsOrigin(2) == True:
            Counter += 1

        if Counter == 0:
          return False

        self.ID = np.zeros(shape=(Counter), dtype=float)
        self.Origin = np.zeros(shape=(Counter), dtype=float)
        self.X = np.zeros(shape=(Counter), dtype=float)
        self.Y = np.zeros(shape=(Counter), dtype=float)
        self.Z = np.zeros(shape=(Counter), dtype=float)
        self.E = np.zeros(shape=(Counter), dtype=float)
        self.Type = np.zeros(shape=(Counter), dtype=float)

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


  def print(self):
    """
    Print the data
    """

    print("Event ID: {}".format(self.EventID))
    print("  Origin Z: {}".format(self.OriginPositionZ))
    print("  Gamma Energy: {}".format(self.GammaEnergy))
    for h in range(0, len(self.X)):
      print("  Hit {} (origin: {}): type={}, pos=({}, {}, {})cm, E={}keV".format(self.ID[h], self.Origin[h], self.Type[h], self.X[h], self.Y[h], self.Z[h], self.E[h]))
