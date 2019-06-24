###################################################################################################
#
# GRBCreator.py
#
# Copyright (C) by Andreas Zoglauer.
# All rights reserved.
#
# Please see the file License.txt in the main repository for the copyright-notice.
#
###################################################################################################




###################################################################################################


import numpy as np


###################################################################################################


class GRBCreator:
  """
  This is the base class for the GRB creator, storing the binning information
  """


###################################################################################################


  def __init__(self, ResolutionInDegrees):
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

    self.PsiMin = -np.pi
    self.PsiMax = +np.pi
    self.PsiBins = int(360 / ResolutionInDegrees)

    self.ChiMin = 0
    self.ChiMax = np.pi
    self.ChiBins = int(180 / ResolutionInDegrees)

    self.PhiMin = 0
    self.PhiMax = np.pi
    self.PhiBins = int(180 / ResolutionInDegrees)



###################################################################################################



# END
###################################################################################################











