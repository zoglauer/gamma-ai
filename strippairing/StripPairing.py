###################################################################################################
#
# StripPairing.py
#
# Copyright (C) by Devyn Donahue & Andreas Zoglauer
# All rights reserved.
#
# Please see the file License.txt in the main repository for the copyright-notice. 
#  
###################################################################################################

  
###################################################################################################


import ROOT

import array
import sys 
 
import itertools
import permutations

  
###################################################################################################



class StripPairing:
  """
  This class performs strip pairing training. A typical usage would look like this:

  AI = StripPairing("Ling2.seq3.quality.root", "Results", "MLP,BDT", 1000000)
  AI.train()
  AI.test()

  """

  
###################################################################################################

  
  def __init__(self, FileName, OutputPrefix, Layout, MaxEvents):

    self.FileName = FileName
    self.Layout = Layout
    self.OutputPrefix = OutputPrefix
    self.MaxEvents = MaxEvents

    
  
###################################################################################################

  
  def train(self):
     
    # Open the file
    DataFile = ROOT.TFile(self.FileName);
    if DataFile.IsOpen() == False:
      print("Error: Opening DataFile")
      sys.exit()

    # Retrieve data from file
    DataTree = DataFile.Get("StripPairing");
    if DataTree == 0:
      print("Error: Reading data tree from root file")
      sys.exit()

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
     
     
     
    # The output file
    ResultsFileName = self.OutputPrefix + ".root"
    ResultsFile = ROOT.TFile(ResultsFileName, "RECREATE")

    # Create the Factory, responible for training and evaluation
    Factory = ROOT.TMVA.Factory("TMVARegression", ResultsFile, "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Regression")

    # Create the data loader - give it the name of the output directory
    DataLoader = ROOT.TMVA.DataLoader(self.OutputPrefix)

    IgnoredBranches = [ 'SimulationID' ]  
    Branches = DataTree.GetListOfBranches()

    # We need to add everything we do not use as spectators, otherwise we do not have access after the training! (I consider this a ROOT bug!)
    for Name in IgnoredBranches:
      DataLoader.AddSpectator(Name, "F")

    # Add the input variables
    for B in list(Branches):
      if not B.GetName() in IgnoredBranches:
        if not B.GetName().startswith("Result"):
          DataLoader.AddVariable(B.GetName(), "F")

    # Add the target variables:
    DataLoader.AddTarget("ResultNumberOfInteractions", "F")
    DataLoader.AddTarget("ResultUndetectedInteractions", "F")
    for B in list(Branches):
      if B.GetName().startswith("ResultInteraction"):
        DataLoader.AddTarget(B.GetName(), "F")


    # Add the regressions tree with weight = 1.0
    DataLoader.AddRegressionTree(DataTree, 1.0);


    # Random split between training and test data
    Cut = ROOT.TCut("")
    DataLoader.PrepareTrainingAndTestTree(Cut, "SplitMode=random:!V");

    # Book a multi-layer perceptron 
    Parameters = ROOT.TString()
    Parameters += "!H:V:VarTransform=Norm:NeuronType=tanh:NCycles=200000:HiddenLayers=" 
    Parameters += self.Layout 
    Parameters += ":TestRate=6:TrainingMethod=BFGS:Sampling=0.3:SamplingEpoch=0.8:ConvergenceImprove=1e-8:ConvergenceTests=15:!UseRegulator"
    Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kMLP, "MLP", Parameters);     

    # Train, test, and evaluate internally
    Factory.TrainAllMethods()
    Factory.TestAllMethods()
    Factory.EvaluateAllMethods()



###################################################################################################

  
  def test(self):

     
    # Open the file
    DataFile = ROOT.TFile(self.FileName);
    if DataFile.IsOpen() == False:
      print("Error: Opening DataFile")
      sys.exit()

    # Read data tree
    DataTree = DataFile.Get("StripPairing");
    if DataTree == 0:
      print("Error: Reading data tree from root file")
      sys.exit()

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

    IgnoredBranches = [ 'SimulationID' ]  
    Branches = DataTree.GetListOfBranches()


    # Setup the reader:
    Reader = ROOT.TMVA.Reader("!Color:!Silent");    

    # Check the multiplicity
    xStrips = 0
    yStrips = 0
    for B in list(Branches):
      Name = B.GetName()
      if Name.startswith('XStripEnergy'):
        xStrips += 1
      if Name.startswith('YStripEnergy'):
        yStrips += 1
    maxStrips = max(xStrips, yStrips)
        
    VariableMap = {}

    # We need to add everything we do not use as spectators, otherwise we do not have access after the training! (I consider this a ROOT bug!)
    for Name in IgnoredBranches:
      VariableMap[Name] = array.array('f', [0])
      DataTree.SetBranchAddress(Name, VariableMap[Name])
      Reader.AddSpectator(Name, VariableMap[Name])


    # Add the input variables
    for B in list(Branches):
      if not B.GetName() in IgnoredBranches:
        if not B.GetName().startswith("Result"):
          VariableMap[B.GetName()] = array.array('f', [0])
          Reader.AddVariable(B.GetName(), VariableMap[B.GetName()])
          DataTree.SetBranchAddress(B.GetName(), VariableMap[B.GetName()])
          print("Added: " + B.GetName())


    # Add the target variables:
    VariableMap["ResultNumberOfInteractions"] = array.array('f', [0])
    DataTree.SetBranchAddress("ResultNumberOfInteractions", VariableMap["ResultNumberOfInteractions"])

    VariableMap["ResultUndetectedInteractions"] = array.array('f', [0])
    DataTree.SetBranchAddress("ResultUndetectedInteractions", VariableMap["ResultUndetectedInteractions"])


    for B in list(Branches):
      if B.GetName().startswith("ResultInteraction"):
        VariableMap[B.GetName()] = array.array('f', [0])
        DataTree.SetBranchAddress(B.GetName(), VariableMap[B.GetName()])


    FileName = ROOT.TString(self.OutputPrefix)
    FileName += "/weights/TMVARegression_MLP.weights.xml"
    Reader.BookMVA("MLP", FileName)

    # Intialize event counters
    NEvents = 0
    NGoodIdentifiedAsGood = 0
    NGoodIdentifiedAsBad = 0
    NBadIdentifiedAsGood = 0
    NBadIdentifiedAsBad = 0
    NGoodEventsTS = 0
    
    NCorrectlyPaired = 0
    NIncorrectlyPaired = 0
    NTooComplex = 0
    
    # Create histograms of the test statistic values:
        
    # keeps objects otherwise removed by garbage collected in a list
    ROOTSaver = []

    # create a new 2D histogram with fine binning
    HistGood = ROOT.TH1F("HistGood", "", 200, 0, 200)
    HistGood.SetLineColor(ROOT.kGreen)
    HistGood.SetXTitle("Test statistics value")
    HistGood.SetYTitle("counts")
    HistBad = ROOT.TH1F("HistBad", "", 200, 0, 200)
    HistBad.SetLineColor(ROOT.kRed)
    HistBad.SetXTitle("Test statistics value")
    HistBad.SetYTitle("counts")

    # Read simulated the events
    for x in range(0, min(10000, DataTree.GetEntries())):
      DataTree.GetEntry(x)
      
      NEvents += 1
      
      print("\nSimulation ID: " + str(int(VariableMap["SimulationID"][0])) + ":")
      
      Result = Reader.EvaluateRegression("MLP")  
      
      NumberOfSimulatedInteractions = int(VariableMap["ResultNumberOfInteractions"][0])
      print("# IAs:      " + str(VariableMap["ResultNumberOfInteractions"][0]) + " vs. " + str(Result[0])) 
      print("Undetected: " + str(VariableMap["ResultUndetectedInteractions"][0]) + " vs. " + str(Result[1])) 
      
      for B in list(Branches):
        Name = B.GetName()
        if Name.startswith('XStripEnergy'):
          print("{}: {}".format(Name, VariableMap[Name][0]))
      for B in list(Branches):
        Name = B.GetName()
        if Name.startswith('YStripEnergy'):
          print("{}: {}".format(Name, VariableMap[Name][0]))
    

      ResultInteractions = []
      # Check to see if result interactions are good or bad
      r = 2
      IsCorrectlyPaired = True
      IsGoodThreshold = 0.3
      NumberOfIdentifiedInteractions = 0
      for B in list(Branches):
        Name = B.GetName()
        if Name.startswith("ResultInteraction"):
          ResultInteractions.append(VariableMap[Name][0])
          print("{}: {} vs. {}".format(Name, VariableMap[Name][0], Result[r]))
          
          # If the difference between the input (0 or 1) is larger than the threshold, than we have not identified the event 
          if abs(VariableMap[Name][0] - Result[r]) > IsGoodThreshold:
            IsCorrectlyPaired = False
          if abs(VariableMap[Name][0] - Result[r]) > IsGoodThreshold and abs(VariableMap[Name][0] - Result[r]) < 0.5:
            IsUndecided = True
            
          if Result[r] > 1 - IsGoodThreshold:
            NumberOfIdentifiedInteractions += 1
          
          r += 1

      if IsCorrectlyPaired == True:
        NCorrectlyPaired += 1
        print(" ---> Correctly paired")         
      else:
        NIncorrectlyPaired += 1
        print(" ---> Incorrectly paired")
        if IsUndecided == True:
          if NumberOfSimulatedInteractions > maxStrips:
            NTooComplex += 1
            print(" -----> Too complex")
          

      # Make list of X and Y strip energies
      XStripList = []
      YStripList = []
      for B in Branches:
        if B.GetName().startswith("XStripEnergy"):
          XStripList.append(VariableMap[B.GetName()][0])
        elif B.GetName().startswith("YStripEnergy"):
          YStripList.append(VariableMap[B.GetName()][0])

      # Generate comparison array fromm ResultsInteraction
      idealinteractions = []
      for B in Branches:
        if B.GetName().startswith("ResultInteraction"):
          idealinteractions.append(VariableMap[B.GetName()][0])

      NX = len(XStripList)
      NY = len(YStripList)

      Result = permutations.CreateStripCombinations(NX, NY)
            
      print(Result)

      # Make the test statistic
      Ts = []
      for grouping in Result:
        N = (len(grouping))
        total = 0
        for pair in grouping:
          x = pair[0]
          y = pair[1]
          total = total + (XStripList[x]-YStripList[y])**2
        Ts.append(total * (1/N))
    
      # Find the minimum from the test statistic
      import numpy as np
      index_min = np.argmin(Ts)

      RITest = np.zeros(len(ResultInteractions)) 
      
      # If it is correct, change it to a 1
      for pair in Result[index_min]:
        x = pair[0]
        y = pair[1]
        index = x + (y*NX)
        RITest[index] = 1

      print("From sim:")
      print(ResultInteractions)
      print("From test statistic")
      print(RITest)  

      if np.all(ResultInteractions == RITest):
        NGoodEventsTS += 1
        HistGood.Fill(Ts[index_min])
        print("Good event from TS!")
      else:
        HistBad.Fill(Ts[index_min])
        print("Bad event from TS!")        

    
    # create a new TCanvas
    ROOTSaver.append(ROOT.TCanvas())
    HistGood.Draw()
    HistBad.Draw("SAME")


    print("\nResult:")
    print("All events: " + str(NEvents))
    print("Number of correctly paired: {} - {}%".format(NCorrectlyPaired, 100.0 * NCorrectlyPaired / NEvents))
    print("Number of incorrectly paired: {} - {}%".format(NIncorrectlyPaired, 100.0 * NIncorrectlyPaired / NEvents))
    print("Number of too complex: {} - {}%".format(NTooComplex, 100.0 * NTooComplex / NEvents))
    print("Good events test statistic: " + str(NGoodEventsTS) +  " (" + str(100.0 * (NGoodEventsTS) / NEvents) + "%)")

    # prevent Canvases from closing
    #wait()
    #print("Close the ROOT window via File -> Close!")
    #ROOT.gApplication.Run()
   


# END  
###################################################################################################


