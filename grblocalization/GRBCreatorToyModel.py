###################################################################################################
#
# GRBCreatorToyModel.py
#
# Copyright (C) by Andreas Zoglauer.
# All rights reserved.
#
# Please see the file License.txt in the main repository for the copyright-notice.
#
###################################################################################################




###################################################################################################


import numpy as np
import random
import sys
import math
from GRBCreator import GRBCreator

import ROOT as M
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")


###################################################################################################


class GRBCreatorToyModel(GRBCreator):
  """
  This class represents GRB creator which uses a toy model
  """


###################################################################################################


  def __init__(self, ResolutionInDegrees, NoiseInDegreesInSigma):
    """
    The default constructor for class EventClustering

    Attributes
    ----------
    ResolutionInDegrees: Float 
      The resolution in degrees of the data space
    NoiseInSigma: Float
      The amount the source data will be noised in degree 

    """

    GRBCreator.__init__(self, ResolutionInDegrees)
    
    self.NoiseInRadiansInSigma = math.radians(NoiseInDegreesInSigma)


###################################################################################################


  def KleinNishina(self, Ei, phi):
    if Ei <= 0:
      print("Error: Invalid input: Ei <= 0")
      return 0
    if phi < 0 or phi > math.pi:
      print("Error: Invalid input: phi < 0 or phi > math.pi")
      return 0

    Radius = 2.8E-15
    E0 = 510.998910

    sinphi = math.sin(phi)
    Eg = -E0*Ei/(math.cos(phi)*Ei-Ei-E0)

    return 0.5*Radius*Radius*Eg*Eg/Ei/Ei*(Eg/Ei+Ei/Eg-sinphi*sinphi)*sinphi


###################################################################################################


  def ComptonScatterAngle(self, Eg, Ee):
    E0 = 510.998910
    Value = 1 - E0 * (1.0/Eg - 1.0/(Ee + Eg))

    if Value <= -1 or Value >= 1:
      print("Error: Invalid input: Value <= -1 or Value >= 1")
      return 0

    return math.acos(Value)


###################################################################################################


  def Create(self, Ei, Rotation):

    # Simulate the gamma ray according to Butcher & Messel: Nuc Phys 20(1960), 15

    Ei_m = Ei / 510.998910

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
  
    CosTeta = 1.0 - OneMinusCosTheta; 

    # Set the new photon parameters --- direction is random since we didn't give a start direction

    Theta = np.arccos(1 - 2*random.random()) # Compton scatter angle since on axis
    Phi = 2.0 * np.pi * random.random();   

    Dg = M.MVector()
    Dg.SetMagThetaPhi(1.0, Theta, Phi) 
    Dg = Rotation * Dg

    Chi = Dg.Theta()
    Psi = Dg.Phi()

    Eg = Epsilon*Ei
    Ee = Ei - Eg
  
    #print(Psi, Chi, Theta, Eg+Ee)
  
    return Chi, Psi, Theta, Eg+Ee


###################################################################################################


  # Dummy noising of the data
  def Noise(self, Chi, Psi, Phi):
    
    NoisedChi = sys.float_info.max
    while NoisedChi < 0 or NoisedChi > math.pi:
      NoisedChi = np.random.normal(Chi, self.NoiseInRadiansInSigma)
      #print("Chi: {} {}".format(Chi, NoisedChi))

    NoisedPsi = sys.float_info.max
    while NoisedPsi < -math.pi or NoisedPsi > math.pi:
      NoisedPsi = np.random.normal(Psi, self.NoiseInRadiansInSigma)
      #print("Psi {} {}".format(Psi, NoisedPsi))

    NoisedPhi = sys.float_info.max
    while NoisedPhi < 0 or NoisedPhi > math.pi:
      NoisedPhi = np.random.normal(Phi, self.NoiseInRadiansInSigma)
      #print("Phi {} {}".format(Phi, NoisedPhi))

    return NoisedChi, NoisedPsi, NoisedPhi


###################################################################################################


  def createOneSourceDataSet(self, Rotation):
    """
    Fill the data from the toy model creator

    Return
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
    

    Chi, Psi, Phi, Energy = self.Create(511, Rotation)
    #print("{}, {}, {}".format(Chi, Psi, Phi))
  
    if self.NoiseInRadiansInSigma > 0:
      Chi, Psi, Phi = self.Noise(Chi, Psi, Phi)

    ChiBin = (int) (((Chi - self.ChiMin) / (self.ChiMax - self.ChiMin)) * self.ChiBins)
    PsiBin = (int) (((Psi - self.PsiMin) / (self.PsiMax - self.PsiMin)) * self.PsiBins)
    PhiBin = (int) (((Phi - self.PhiMin) / (self.PhiMax - self.PhiMin)) * self.PhiBins)
    
    Index = PsiBin*self.ChiBins*self.PhiBins + ChiBin*self.PhiBins + PhiBin
    
    return Index


###################################################################################################


  def createOneBackgroundDataSet(self):
    """
    Fill the data from the toy model creator

    Return
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
    
    ChiBin = random.randint(0, self.ChiBins-1)
    PsiBin = random.randint(0, self.PsiBins-1)
    PhiBin = random.randint(0, self.PhiBins-1)

    Index = PsiBin*self.ChiBins*self.PhiBins + ChiBin*self.PhiBins + PhiBin

    return Index










