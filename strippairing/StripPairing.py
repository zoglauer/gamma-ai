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

    self.UseOnlyGoodEvents = False
    self.NormalizeEnergies = False
    
    self.NXStrips = 0
    self.NYStrips = 0
    
  
###################################################################################################

  
  def getData(self):
    
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

    # All branches
    Branches = list(DataTree.GetListOfBranches())

    # Map all columns
    VariableMap = {}
    for B in Branches:
      Name = B.GetName()
      VariableMap[Name] = array.array('f', [0])
      DataTree.SetBranchAddress(Name, VariableMap[Name])

    # Retrieve the number of triggered strips
    self.NXStrips = 0
    self.NYStrips = 0
    for B in Branches:
      Name = B.GetName()
      if Name.startswith('XStripEnergy'):
        self.NXStrips += 1
      if Name.startswith('YStripEnergy'):
        self.NYStrips += 1

    # Create a new tree
    NewTree = DataTree.CloneTree(0);
    NewTree.SetDirectory(0);

    # Copy data to the new tree
    EntryIndex = 0
    NewEntries = 0
    while NewEntries < self.MaxEvents and EntryIndex < DataTree.GetEntries():
      DataTree.GetEntry(EntryIndex)

      if self.NormalizeEnergies == True:
        MinEnergy = sys.float_info.max
        MaxEnergy = 0
        
        for B in Branches:
          Name = B.GetName()
          if 'StripEnergy' in Name:
            if VariableMap[Name][0] < MinEnergy:
              MinEnergy = VariableMap[Name][0]
            if VariableMap[Name][0] > MaxEnergy:
              MaxEnergy = VariableMap[Name][0]
    
        if MinEnergy > MaxEnergy:
          print("ERROR: Unable to determine minimum ({0}) and maximum energy ({1}). Aborting...".format(MinEnergy, MaxEnergy))
          sys.exit(1) 
          
        for B in Branches:
          Name = B.GetName()
          if 'StripEnergy' in Name:
            if MinEnergy != MaxEnergy:      
              VariableMap[Name][0] = (VariableMap[Name][0] - MinEnergy) / (MaxEnergy - MinEnergy)
            else:
              VariableMap[Name][0] = 0.0      

      if self.UseOnlyGoodEvents == False or VariableMap['ResultNumberOfInteractions'][0] == maxStrips: 
        NewTree.Fill()
        NewEntries += 1
      EntryIndex += 1
        

    return NewTree
    
  
###################################################################################################

  
  def train(self):
     
    # Part 1: Get all the data
     
    DataTree = self.getData()

    print("Analyzing {} events...".format(DataTree.GetEntries()))


    # Part 2: Setup TMVA

    # Initialize TMVA
    ROOT.TMVA.Tools.Instance()
     
    # The output file
    ResultsFileName = self.OutputPrefix + ".x" + str(self.NXStrips) + ".y" + str(self.NYStrips) + ".root"
    ResultsFile = ROOT.TFile(ResultsFileName, "RECREATE")

    # Create the Factory, responible for training and evaluation
    Factory = ROOT.TMVA.Factory("TMVARegression", ResultsFile, "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Regression")

    # Create the data loader - give it the name of the output directory
    DataLoader = ROOT.TMVA.DataLoader(self.OutputPrefix + ".x" + str(self.NXStrips) + ".y" + str(self.NYStrips))

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

    # Add the target variables - ResultNumberOfInteractions is only one if we do not train on only good events:
    if self.UseOnlyGoodEvents == False:
      DataLoader.AddTarget("ResultNumberOfInteractions", "F")
    else:
      DataLoader.AddSpectator("ResultNumberOfInteractions", "F")
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
    Parameters += ":TestRate=6:TrainingMethod=BFGS:Sampling=0.3:SamplingEpoch=0.8:ConvergenceImprove=1e-8:ConvergenceTests=30:!UseRegulator"
    Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kMLP, "MLP", Parameters);     

    # Train, test, and evaluate internally
    Factory.TrainAllMethods()
    Factory.TestAllMethods()
    Factory.EvaluateAllMethods()



###################################################################################################

  
  def test(self):
     
    # Part 1: Get all the data

    DataTree = self.getData()
    maxStrips = max(self.NXStrips, self.NYStrips)

    print("Analyzing {} events...".format(DataTree.GetEntries()))
    

    # Part 2: Setup TMVA

    # Initialize TMVA
    ROOT.TMVA.Tools.Instance()

    IgnoredBranches = [ 'SimulationID' ]  
    Branches = DataTree.GetListOfBranches()


    # Setup the reader:
    Reader = ROOT.TMVA.Reader("!Color:!Silent");    

        
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
    if self.UseOnlyGoodEvents == True:
      Reader.AddSpectator("ResultNumberOfInteractions", VariableMap["ResultNumberOfInteractions"])

    VariableMap["ResultUndetectedInteractions"] = array.array('f', [0])
    DataTree.SetBranchAddress("ResultUndetectedInteractions", VariableMap["ResultUndetectedInteractions"])

    for B in list(Branches):
      if B.GetName().startswith("ResultInteraction"):
        VariableMap[B.GetName()] = array.array('f', [0])
        DataTree.SetBranchAddress(B.GetName(), VariableMap[B.GetName()])


    FileName = ROOT.TString(self.OutputPrefix + ".x" + str(self.NXStrips) + ".y" + str(self.NYStrips))
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
    NIncorrectlyIdentified = 0
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
    

      ResultInteractions = []
      # Check to see if result interactions are good or bad
      Index = 0
      StartIndex = 2
      if self.UseOnlyGoodEvents == True:
        StartIndex = 1
        
      IsCorrectlyPaired = True
      IsGoodThreshold = 0.49
      NumberOfIdentifiedInteractions = 0
      for B in list(Branches):
        Name = B.GetName()
        if Name.startswith("ResultInteraction"):
          ResultInteractions.append(VariableMap[Name][0])
          
          # If the difference between the input (0 or 1) is larger than the threshold, than we have not identified the event 
          if abs(VariableMap[Name][0] - Result[StartIndex + Index]) > IsGoodThreshold:
            IsCorrectlyPaired = False
            
          if Result[StartIndex + Index] > 1 - IsGoodThreshold:
            NumberOfIdentifiedInteractions += 1
          
          Index += 1


      # Statistics & printing
      if IsCorrectlyPaired == True:
        NCorrectlyPaired += 1
        print(" ---> Correctly paired")         
      else:
        if Result[0] > maxStrips + 0.25:
          NTooComplex += 1
          print(" -----> Too complex")
        else:
          NIncorrectlyIdentified += 1
          print(" ---> Incorrectly paired")
        
      print("Number of IAs:   {} vs. {}".format(VariableMap["ResultNumberOfInteractions"][0], Result[0])) 
      print("Undetected:      {} vs. {}".format(VariableMap["ResultUndetectedInteractions"][0], Result[1])) 
        
      for B in list(Branches):
        Name = B.GetName()
        if Name.startswith('XStripEnergy'):
          print("{}: {}".format(Name, VariableMap[Name][0]))
      for B in list(Branches):
        Name = B.GetName()
        if Name.startswith('YStripEnergy'):
          print("{}: {}".format(Name, VariableMap[Name][0]))
        
      Index = 0
      for B in list(Branches):
        Name = B.GetName()
        if Name.startswith("ResultInteraction"):
          print("{}: {} vs. {}".format(Name, VariableMap[Name][0], Result[StartIndex + Index]))
          Index += 1
          
      
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
            
      #print(Result)

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

      #if IsCorrectlyPaired == False:
      print("From sim:")
      print(ResultInteractions)
      print("From test statistic")
      print(RITest)  

      if np.all(ResultInteractions == RITest):
        NGoodEventsTS += 1
        HistGood.Fill(Ts[index_min])
        if IsCorrectlyPaired == False:
          print("Good event from TS!")
      else:
        HistBad.Fill(Ts[index_min])
        if IsCorrectlyPaired == False:
          print("Bad event from TS!")   

    
    # create a new TCanvas
    ROOTSaver.append(ROOT.TCanvas())
    HistGood.Draw()
    HistBad.Draw("SAME")


    print("\nResult:")
    print("All events: " + str(NEvents))
    print("Number of correctly paired: {} - {}%".format(NCorrectlyPaired, 100.0 * NCorrectlyPaired / NEvents))
    print("Number of too complex: {} - {}%".format(NTooComplex, 100.0 * NTooComplex / NEvents))
    print("Number of correctly identified: {} - {}%".format((NCorrectlyPaired + NTooComplex) , 100.0 * (NCorrectlyPaired + NTooComplex)  / NEvents))
    print("Number of incorrectly identified: {} - {}%".format(NIncorrectlyIdentified, 100.0 * NIncorrectlyIdentified / NEvents))
    print("Good events test statistic: " + str(NGoodEventsTS) +  " (" + str(100.0 * (NGoodEventsTS) / NEvents) + "%)")

    return True, 100.0 * (NCorrectlyPaired + NTooComplex) / NEvents, 100.0 * NIncorrectlyIdentified / NEvents

    # prevent Canvases from closing
    #wait()
    #print("Close the ROOT window via File -> Close!")
    #ROOT.gApplication.Run()
   


# END  
###################################################################################################


