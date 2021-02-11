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
random.seed(0) # added this line for debugging
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
    self.Type   = np.zeros(shape=(self.MaxHits), dtype=np.dtype('U2'))

    self.Acceptance = "egpb"


###################################################################################################


  def setAcceptance(self, Acceptance):
    """
    Set which track types to accept:
    e: electron
    g: gamma
    p: positron
    b: bremsstrahlung
    """
    self.Acceptance = Acceptance


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
        self.Type[0] = "eg"
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

    Debug = False

    self.ID = SimEvent.GetID()


    # Clusterize adjacent strip hits
    SimEvent.CreateClusters()
    Hits = []
    for c in range(0, SimEvent.GetNClusters()):
      HT = SimEvent.GetClusterAt(c).CreateHT()
      M.SetOwnership(HT, True) # Python needs ownership of the event in order to delete it
      Hits.append(HT)


    SimEvent.DeleteAllHTs()

    for h in range(0, len(Hits)):
      SimEvent.AddHT(Hits[h])

    # Only pick good events
    if SimEvent.GetNIAs() <= 3:
      if Debug == True: print("Event {} rejected: Not enough IAs: {}".format(self.ID, SimEvent.GetNIAs()))
      return False

    if SimEvent.GetNHTs() < 2:
      if Debug == True: print("Event {} rejected: Not enough hits: {}".format(self.ID, SimEvent.GetNHTs()))
      return False

    if SimEvent.GetIAAt(1).GetProcess() != M.MString("COMP"):
      if Debug == True: print("Event {} rejected: First interaction not Compton: {}".format(self.ID, SimEvent.GetIAAt(1).GetProcess().Data()))
      return False

    if SimEvent.GetIAAt(1).GetDetectorType() != 1 and SimEvent.GetIAAt(1).GetDetectorType() != 3:
      if Debug == True: print("Event {} rejected: First interaction not in strip detector: {}".format(self.ID, SimEvent.GetIAAt(1).GetDetectorType()))
      return False

    if SimEvent.GetIAAt(2).GetDetectorType() == 1:
      if Debug == True: print("Event {} rejected: Second interaction in tracker".format(self.ID))
      return False

    if SimEvent.GetNPMs() > 0:
      if Debug == True: print("Event {} rejected: Energy deposits in passive material found".format(self.ID))
      return False

    if SimEvent.IsIACompletelyAbsorbed(1, 10.0, 2.0) == False:
      if Debug == True: print("Event {} rejected: Not completely absorbed".format(self.ID))
      return False

    if SimEvent.GetNGRs() > 0:
      if Debug == True: print("Event {} rejected: Guard ring vetoes".format(self.ID))
      return False

    for i in range(0, SimEvent.GetNIAs()):
      if SimEvent.GetIAAt(i).GetProcess() == M.MString("PAIR"):
        if Debug == True: print("Event {} rejected: Pair interaction found".format(self.ID))
        return False
      if SimEvent.GetIAAt(i).GetProcess() == M.MString("BREM"):
        if Debug == True: print("Event {} rejected: Bremsstrahlung found".format(self.ID))
        return False
      if SimEvent.GetIAAt(i).GetProcess() == M.MString("RAYL"):
        if Debug == True: print("Event {} rejected: Rayleigh interaction found".format(self.ID))
        return False
      if SimEvent.GetIAAt(i).GetProcess() == M.MString("ESCP"):
        if Debug == True: print("Event {} rejected: Particle escape found".format(self.ID))
        return False


    Counter = SimEvent.GetNHTs()

    # Origin = np.zeros(shape=(Counter), dtype=int)
    self.Origin = np.zeros(shape=(Counter), dtype=int)
    self.ID = np.zeros(shape=(Counter), dtype=int)
    self.X = np.zeros(shape=(Counter), dtype=float)
    self.Y = np.zeros(shape=(Counter), dtype=float)
    self.Z = np.zeros(shape=(Counter), dtype=float)
    self.E = np.zeros(shape=(Counter), dtype=float)
    self.Type = np.zeros(shape=(Counter), dtype=np.dtype('U2'))

    self.OriginPositionX = SimEvent.GetIAAt(1).GetPosition().X()
    self.OriginPositionY = SimEvent.GetIAAt(1).GetPosition().Y()
    self.OriginPositionZ = SimEvent.GetIAAt(1).GetPosition().Z()

    IsOriginIncluded = False

    ZMin = 1000
    ZMax = -1000

    Counter = 0
    for i in range(0, SimEvent.GetNHTs()):
      Previous, TrackType = self.previousHTandType(SimEvent, i)
      self.Origin[Counter] = Previous+1
      self.ID[Counter] = i+1
      self.X[Counter] = SimEvent.GetHTAt(i).GetPosition().X()
      self.Y[Counter] = SimEvent.GetHTAt(i).GetPosition().Y()
      self.Z[Counter] = SimEvent.GetHTAt(i).GetPosition().Z()
      self.E[Counter] = SimEvent.GetHTAt(i).GetEnergy()
      self.Type[Counter] = TrackType

      if self.Z[Counter] < ZMin:
        ZMin = self.Z[Counter]

      if self.Z[Counter] > ZMax:
        ZMax = self.Z[Counter]

      if math.fabs(self.Z[Counter] - self.OriginPositionZ) < 0.1:
        IsOriginIncluded = True

      Counter += 1

    if IsOriginIncluded == False:
      return False

    # make sure there is only an eg if really an electron is emerging
    for i in range(0, Counter):
      if self.Type[i] == "eg":
        FoundTrack = False
        for j in range(0, Counter):
          if self.Origin[j] == self.ID[i] and self.Type[j] == "e":
            FoundTrack = True
            break;
        if FoundTrack == False:
          self.Type[i] = "g"


    # If we have no electron track, reject the event
    if "e" in self.Acceptance:
      FoundTrack = False
      for i in range(0, Counter):
        if self.Type[i] == "e":
          FoundTrack = True
          break
      if FoundTrack == False:
        if Debug == True: print("Event {} rejected: No electron track".format(self.ID))
        return False

    # If we dont't have a "b" in acceptance reject all events with a bremsstrahlung hit
    if not "p" in self.Acceptance:
      for i in range(0, Counter):
        if "p" in self.Type[i]:
          if Debug == True: print("Event {} rejected: Not accepting events with positrons".format(self.ID))
          return False

    # If we dont't have a "b" in acceptance reject all events with a bremsstrhlung hit
    if not "b" in self.Acceptance:
      for i in range(0, Counter):
        if "b" in self.Type[i]:
          if Debug == True: print("Event {} rejected: Not accepting hits with bremstrahlung".format(self.ID))
          return False

    # If we don't have "e" in acceptance reject all events with a track
    if not "e" in self.Acceptance:
      for i in range(0, Counter):
        if self.Type[i] == "e":
          if Debug == True: print("Event {} rejected: Not accepting hits with electron tracks".format(self.ID))
          return False

    # If we don't have "g" in acceptance remove all hits with just a gamma interaction
    if not "g" in self.Acceptance:
      ToRemove = []
      for i in range(0, Counter):
        if self.Type[i] == "g":
          ToRemove.append(i)
      self.Origin = np.delete(self.Origin, ToRemove)
      self.ID = np.delete(self.ID, ToRemove)
      self.X = np.delete(self.X, ToRemove)
      self.Y = np.delete(self.Y, ToRemove)
      self.Z = np.delete(self.Z, ToRemove)
      self.E = np.delete(self.E, ToRemove)
      self.Type = np.delete(self.Type, ToRemove)

    self.unique = len(np.unique(self.Z))
    filter = 4

    # if "g" in self.Acceptance:
    #     if (self.unique != filter):
    #         return False

    if Debug == True:
      print(SimEvent.ToSimString().Data())
      self.print()

    return True


###################################################################################################


  def previousHTandType(self, SimEvent, ID):
    """
    Return the previous HT ID given the HT ID in the SimEvent, -1 if there is none
    """

    # If there is a hit with the same Origin, but earlier up in the list, this one is the earlier

    Type = "eg"
    PreviousHitID = -1

    SmallestOriginID = SimEvent.GetHTAt(ID).GetSmallestOrigin()
    #print("SmallestOriginID: {}".format(SmallestOriginID))
    if ID > 0:
      #print("Before range {}".format(range(ID-1, 0, -1)))
      for h in range(ID-1, -1, -1):
        #print(h)
        if SimEvent.GetHTAt(h).IsOrigin(SmallestOriginID) == True:
          PreviousHitID = h
          Type = "e"
          break

    if PreviousHitID >= 0:
      #print("Previous hit for {} in same track {}".format(ID, PreviousHitID))
      return PreviousHitID, Type

    # Now check the origin one up, if they have the same origin ID. If this is the case the one up is the previous IA and check for its HTs
    if SmallestOriginID > 1:
      OriginID = SmallestOriginID
      while OriginID > 1:
        IAOriginID = SimEvent.GetIAAt(OriginID-1).GetOriginID()
        IAOriginIDUp = SimEvent.GetIAAt(OriginID-2).GetOriginID()

        #print("Origins here and up: {} {}".format(IAOriginID, IAOriginIDUp))
        if IAOriginID == IAOriginIDUp:
          for h in range(0, SimEvent.GetNHTs()):
            if SimEvent.GetHTAt(h).GetSmallestOrigin() == SimEvent.GetIAAt(OriginID-2).GetId():
              PreviousHitID = h
              Type = self.getType(SimEvent.GetIAAt(OriginID-2).GetProcess().Data(), SimEvent.GetIAAt(OriginID-2).GetSecondaryParticleID())
              break

          if PreviousHitID >= 0:
            #print("Previous hit for {} in main IA sequence {}".format(ID, PreviousHitID))
            return PreviousHitID, Type

        OriginID -= 1


    # Now go through IA's this one originated from, if they have hits, if yes it is the Hit with the SMALLEST ID
    OriginID = SmallestOriginID
    while True:
      OriginID = SimEvent.GetIAById(OriginID).GetOriginID()
      #print("Origin: {}".format(OriginID))
      if OriginID > 1:
        for h in range(0, SimEvent.GetNHTs()):
          if SimEvent.GetHTAt(h).GetSmallestOrigin() == OriginID:
            PreviousHitID = h
            Type = self.getType(SimEvent.GetIAById(OriginID).GetProcess().Data(), SimEvent.GetIAById(OriginID).GetSecondaryParticleID())
            break
      else:
        break

      if PreviousHitID >= 0:
        #print("Previous hit for {} in main IA sequence {}".format(ID, PreviousHitID))
        return PreviousHitID, Type

    # Nothing found
    #print("No previous hit found")
    return -1, "eg"


###################################################################################################


  def getType(self, Process, ParticleID):
    if Process == "COMP":
      return "eg"
    elif Process == "BREM":
      return "e"
    elif Process == "PHOT":
      return "g"
    elif Process == "PAIR" and ParticleID == 3:
      return "e"
    elif Process == "PAIR" and ParticleID == 2:
      return "p"
    else:
      return "?"



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
