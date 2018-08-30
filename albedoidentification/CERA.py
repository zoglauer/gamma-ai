
###################################################################################################
#
# EnergyLoss.py
#
# Copyright (C) by Andreas Zoglauer, Jasper Gan, & Joan Zhu.
# All rights reserved.
#
# Please see the file License.txt in the main repository for the copyright-notice. 
#  
###################################################################################################




###################################################################################################


import ROOT
import array
import sys


###################################################################################################


class CERA:

  """
  This class performs classification on evaulation of whether isReconstructable and isAbsortbed. 
  A typical usage would look like this:

  AI = EnergyLossIdentification("Ling2.seq3.quality.root", "Results", "MLP,BDT", 1000000)
  AI.train()
  AI.test()"""

  def __init__(self, Filename, Output, Algorithm, MaxEvents, Quality):
    self.Filename = Filename
    self.OutputPrefix = Output
    self.Algorithms = Algorithm
    self.MaxEvents = MaxEvents
    self.Quality = Quality


###################################################################################################


  def train(self):
    """
    Switch between the various machine-learning libraries based on self.Algorithm
    """ 
    
    if self.Algorithms.startswith("TMVA:"):
      self.trainTMVAMethods()
    # elif self.Algorithms.startswith("SKL:"):
    #   self.trainSKLMethods()
    else:
      print("ERROR: Unknown algorithm: {}".format(self.Algorithms))
    
    return
  
  
###################################################################################################


  def trainTMVAMethods(self):
    """
    Main training function 
    
    Returns
    -------
    bool
      True is everything went well, False in case of an error 
      
    """
    # Open the file
    DataFile = ROOT.TFile(self.Filename)
    if DataFile.IsOpen() == False:
      print("Error opening data file")
      return False

    # Get the data tree
    DataTree = DataFile.Get(self.Quality)
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
      ResultsFile = ROOT.TFile(FullPrefix + ".root","RECREATE")

      Factory = ROOT.TMVA.Factory("TMVAClassification", fout,
                                  ":".join([
                                      "!V",
                                      "!Silent",
                                      "Color",
                                      "DrawProgressBar",
                                      "Transformations=I;D;P;G,D",
                                      "AnalysisType=Classification"]
                                           ))

      DataLoader = ROOT.TMVA.DataLoader(self.OutputPrefix)

      IgnoredBranches = [ 'SimulationID', 'SequenceLength']
      Branches = DataTree.GetListOfBranches()

      for Name in IgnoredBranches:
          DataLoader.AddSpectator(Name, "F")

      for B in list(Branches):
          if not B.GetName() in IgnoredBranches:
              if not B.GetName().startswith("Evaluation"):
                  DataLoader.AddVariable(B.GetName(), "F")

      SignalCut = ROOT.TCut("EvaluationIsReconstructable >= 0.5")
      BackgroundCut = ROOT.TCut("EvaluationIsReconstructable < 0.5")
      DataLoader.SetInputTrees(DataTree, SignalCut, BackgroundCut)

      DataLoader.PrepareTrainingAndTestTree(SignalCut,
                                         BackgroundCut,
                                         ":".join([
                                              "nTrain_Signal=0",
                                              "nTrain_Background=0",
                                              "SplitMode=Random",
                                              "NormMode=NumEvents",
                                              "!V"
                                             ]))

      # Neural Networks
      if 'MLP' in self.Algorithms:
        method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kMLP, "MLP",
            ":".join([
                "H:",
                "!V",
                "NeuronType=tanh",
                "VarTransform=N",
                "NCycles=100",
                "HiddenLayers=2*N,N",
                "TestRate=5",
                "!UseRegulator"
                ]))


      # PDEFoamBoost
      if 'PDEFoamBoost' in self.Algorithms:
        method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kPDEFoam, "PDEFoamBoost",
          ":".join([
            "!H",
            "!V",
            "Boost_Num=30",
            "Boost_Transform=linear",
            "SigBgSeparate=F",
            "MaxDepth=4",
            "UseYesNoCell=T",
            "DTLogic=MisClassificationError",
            "FillFoamWithOrigWeights=F",
            "TailCut=0",
            "nActiveCells=500",
            "nBin=20",
            "Nmin=400",
            "Kernel=None",
            "Compress=T"
            ]))

      # PDERSPCA
      if 'PDERSPCA' in self.Algorithms:
        method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kPDERS, "PDERSPCA",
            ":".join([
                "!H",
                "!V",
                "VolumeRangeMode=Adaptive",
                "KernelEstimator=Gauss",
                "GaussSigma=0.3",
                "NEventsMin=400",
                "NEventsMax=600",
                "VarTransform=PCA"
            ]))

      # Random Forest Boosted Decision Trees
      if 'BDT' in self.Algorithms:
        method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kBDT, "BDT",
                           ":".join([
                               "!H",
                               "!V",
                               "NTrees=850",
                               "nEventsMin=150",
                               "MaxDepth=3",
                               "BoostType=AdaBoost",
                               "AdaBoostBeta=0.5",
                               "SeparationType=GiniIndex",
                               "nCuts=20",
                               "PruneMethod=NoPruning",
                               ]))


      print("Started training")
      Factory.TrainAllMethods()
      Factory.TestAllMethods()
      Factory.EvaluateAllMethods()

      reader = ROOT.TMVA.Reader("!Color:!Silent");
      variablemap = {}

      for name in IgnoredBranches:
        variablemap[name] = array.array('f', [0])
        datatree.SetBranchAddress(name, variablemap[name])
        reader.AddSpectator(name, variablemap[name])

      for b in list(Branches):
        if not b.GetName() in IgnoredBranches:
          if not b.GetName().startswith("Evaluation"):
            variablemap[b.GetName()] = array.array('f', [0])
            reader.AddVariable(b.GetName(), variablemap[b.GetName()])
            datatree.SetBranchAddress(b.GetName(), variablemap[b.GetName()])
            print("Added: " + b.GetName())

      for b in list(Branches):
        if b.GetName().startswith("EvaluationIsReconstructable") or b.GetName().startswith("EvaluationIsCompletelyAborbed"):
          variablemap[b.GetName()] = array.array('f', [0])
          datatree.SetBranchAddress(b.GetName(), variablemap[b.GetName()])

      reader.BookMVA("BDT","Results/weights/TMVAClassification_BDT.weights.xml")

      NEvents = 0
      NGoodEvents = 0

      NLearnedGoodEvents = 0

      NLearnedCorrectEvents = 0

      varx = array.array('f',[0]) #; reader.AddVariable("EvaluationZenithAngle",varx)
      vary = array.array('f',[0]) #; reader.AddVariable("result",vary)

      for x in range(0, min(500, datatree.GetEntries())):
        datatree.GetEntry(x)

        NEvents += 1

        print("\nSimulation ID: " + str(int(variablemap["SimulationID"][0])) + ":")

        result = reader.EvaluateMVA("BDT")
        print(result)
        vary.append(result)

        r = 2
        IsGood = True
        IsGoodThreshold = 0.2

        IsLearnedGood = True
        IsLearnedGoodThreshold = 0.06 # Adjust this as see fit

        for b in list(Branches):
          name = b.GetName()

          if name.startswith("EvaluationIsReconstructable") or name.startswith("EvaluationIsCompletelyAborbed"):
            print(name + " " + str(variablemap[name][0]))
            if not variablemap[name][0]:
              IsGood = False
            if result < IsLearnedGoodThreshold:
              IsLearnedGood = False
            r += 1

        if IsGood == True:
          NGoodEvents += 1
          print(" --> Good event")
        else:
          print(" --> Bad event")

        if (IsLearnedGood == True and IsGood == True) or (IsLearnedGood == False and IsGood == False):
          NLearnedCorrectEvents += 1

      print("\nResult:")
      print("All events: " + str(NEvents))
      print("Good events: " + str(NGoodEvents))
      print("Correctly identified: " + str(NLearnedCorrectEvents / NEvents))

      gcSaver = []

      gcSaver.append(ROOT.TCanvas())

      histo2 = ROOT.TH2F("histo2","",200,-5,5,200,-5,5)

      # loop over the bins of a 2D histogram
      for i in range(1,histo2.GetNbinsX() + 1):
          for j in range(1,histo2.GetNbinsY() + 1):

              # find the bin center coordinates
              varx[0] = histo2.GetXaxis().GetBinCenter(i)
              vary[0] = histo2.GetYaxis().GetBinCenter(j)

              # calculate the value of the classifier
              # function at the given coordinate
              bdtOutput = reader.EvaluateMVA("BDT")

              # set the bin content equal to the classifier output
              histo2.SetBinContent(i,j,bdtOutput)

      gcSaver.append(ROOT.TCanvas())
      histo2.Draw("colz")

      # draw sigma contours around means
      for mean, color in (
          ((1,1), ROOT.kRed), # signal
          ((-1,-1), ROOT.kBlue), # background
          ):

          # draw contours at 1 and 2 sigmas
          for numSigmas in (1,2):
              circle = ROOT.TEllipse(mean[0], mean[1], numSigmas)
              circle.SetFillStyle(0)
              circle.SetLineColor(color)
              circle.SetLineWidth(2)
              circle.Draw()
              gcSaver.append(circle)

      ROOT.TestTree.Draw("BDT>>hSig(22,-1.1,1.1)","classID == 0","goff")  # signal
      ROOT.TestTree.Draw("BDT>>hBg(22,-1.1,1.1)","classID == 1", "goff")  # background

      ROOT.hSig.SetLineColor(ROOT.kRed); ROOT.hSig.SetLineWidth(2)  # signal histogram
      ROOT.hBg.SetLineColor(ROOT.kBlue); ROOT.hBg.SetLineWidth(2)   # background histogram

      # use a THStack to show both histograms
      hs = ROOT.THStack("hs","")
      hs.Add(ROOT.hSig)
      hs.Add(ROOT.hBg)

      # show the histograms
      gcSaver.append(ROOT.TCanvas())
      hs.Draw()

      # prevent Canvases from closing
      print("Close the ROOT window via File -> Close!")
      ROOT.gApplication.Run()

###################################################################################################


  def test(self):
    """
    Main test function
    
    Returns
    -------
    bool
      True is everything went well, False in case of an error 
      
    """
    
    return True
    



# END  
###################################################################################################