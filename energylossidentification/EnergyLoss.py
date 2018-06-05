###################################################################################################
#
# EnergyLoss.py
#
# Copyright (C) by Andreas Zoglauer.
# All rights reserved.
#
# Please see the file License.txt in the main repository for the copyright-notice. 
#  
###################################################################################################


# TODO: Train and test all multiplicities
# TODO: Test performance as a function of energy
# TODO: Test deep neural Networks
# TODO: Test different libraries

  
###################################################################################################


import ROOT
import array
import os
import sys 

  
###################################################################################################


class EnergyLossIdentification:
  """
  This class performs energy loss training. A typical usage would look like this:

  AI = EnergyLoss("EC.maxhits3.eventclusterizer.root", False, "Results", "3*N,N", "MLP", "100000")
  AI.train()
  AI.test()

  """

  
###################################################################################################

  
  def __init__(self, FileName, Output, Algorithm, MaxEvents):
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
    
    self.FileName = FileName
    self.OutputPrefix = Output
    self.Algorithms = Algorithm
    self.MaxEvents = MaxEvents

  
###################################################################################################


  def train(self):

    # Open the file
    DataFile = ROOT.TFile(self.FileName)
    if DataFile.IsOpen() == False:
      print("Error opening data file")
      return False

    # Get the data tree
    DataTree = DataFile.Get("Quality")
    if DataTree == 0:
      print("Error reading data tree from root file")
      return False

    # Limit the number of events:
    if DataTree.GetEntries() > self.MaxEvents:
      print("Reducing source tree size from " + str(DataTree.GetEntries()) + " to " + str(self.MaxEvents) + " (i.e. the maximum set)")
      NewTree = DataTree.CloneTree(0);
      NewTree.SetDirectory(0);
    
      for i in range(0, self.MaxEvents):
        DataTree.GetEntry(i)
        NewTree.Fill()
    
      DataTree = NewTree;


    # Initialize TMVA
    ROOT.TMVA.Tools.Instance()

    FullPrefix = self.OutputPrefix 
    ResultsFile = ROOT.TFile(FullPrefix + ".root", "RECREATE")

    Factory = ROOT.TMVA.Factory("TMVAClassification", ResultsFile, "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification")

    DataLoader = ROOT.TMVA.DataLoader(self.OutputPrefix)

    IgnoredBranches = [ 'SimulationID', 'SequenceLength']
    Branches = DataTree.GetListOfBranches()

    for Name in IgnoredBranches:
      DataLoader.AddSpectator(Name, "F")

    for B in list(Branches):
      if not B.GetName() in IgnoredBranches:
        if not B.GetName().startswith("Evaluation"):
          DataLoader.AddVariable(B.GetName(), "F")

    SignalCut = ROOT.TCut("EvaluationIsCompletelyAbsorbed >= 0.5")
    BackgroundCut = ROOT.TCut("EvaluationIsCompletelyAbsorbed < 0.5")
    DataLoader.SetInputTrees(DataTree, SignalCut, BackgroundCut)

    DataLoader.PrepareTrainingAndTestTree(SignalCut, BackgroundCut, "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V")

    # Neural Networks
    if 'MLP' in self.Algorithms:
      method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kMLP, "MLP", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=100:HiddenLayers=N+10,N-5:TestRate=5:TrainingMethod=BFGS:!UseRegulator")
      #method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kMLP, "MLP", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=100:HiddenLayers=N+10,N-5:TestRate=5:!UseRegulator")
      #method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kMLP, "MLP", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=100:HiddenLayers=N+10,N-5:TestRate=6:TrainingMethod=BFGS:Sampling=0.3:SamplingEpoch=0.8:ConvergenceImprove=1e-6:ConvergenceTests=15:!UseRegulator")


    # PDEFoamBoost
    if 'PDEFoamBoost' in self.Algorithms:
      method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kPDEFoam, "PDEFoamBoost", "!H:!V:Boost_Num=30:Boost_Transform=linear:SigBgSeparate=F:MaxDepth=4:UseYesNoCell=T:DTLogic=MisClassificationError:FillFoamWithOrigWeights=F:TailCut=0:nActiveCells=500:nBin=20:Nmin=400:Kernel=None:Compress=T")

    # PDERSPCA
    if 'PDERSPCA' in self.Algorithms:
      method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kPDERS, "PDERSPCA", "!H:!V:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600:VarTransform=PCA")

    # Random Forest Boosted Decision Trees
    if 'BDT' in self.Algorithms:
      method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kBDT, "BDT", "!H:!V:NTrees=1000:MinNodeSize=1%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.4:SeparationType=CrossEntropy:nCuts=100:PruneMethod=NoPruning")
      #method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kBDT, "BDT", "!H:!V:NTrees=850:nEventsMin=150:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning")
      #method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kBDT, "BDT", "!H:!V:NTrees=1000:nEventsMin=1000:MaxDepth=4:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning")

    # State Vector Machine
    if 'SVM' in self.Algorithms:
      method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kSVM, "SVM", "Gamma=0.25:Tol=0.001:VarTransform=Norm");


    Factory.TrainAllMethods()
    Factory.TestAllMethods()
    Factory.EvaluateAllMethods()
    
    return True

  
###################################################################################################


  def test(self):
    """
    Test the given file
    
    Returns
    -------
    bool
      True is everything went well, False in case of an error 
      
    """
    
    
    



# END  
###################################################################################################
