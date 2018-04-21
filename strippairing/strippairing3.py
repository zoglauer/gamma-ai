
import ROOT
import array
import sys 
 

import itertools
import permutations

class StripPairing:

  def __init__(self, FileName, OutputPrefix, Layout):
    self.FileName = FileName
    self.Layout = Layout
    self.OutputPrefix = OutputPrefix

  def train(self):
     
    # (1) Read the data tree
    DataFile = ROOT.TFile(self.FileName);
    if DataFile.IsOpen() == False:
      print("Error: Opening DataFile")
      sys.exit()

    # TODO: Determine string from file name
    DataTree = DataFile.Get("StripPairing_2_2");
    if DataTree == 0:
      print("Error: Reading data tree from root file")
      sys.exit()

    # Initialize TMVA
    ROOT.TMVA.Tools.Instance()
     

     
    # PART 1: Train the neural network 
     
     
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
    Parameters += "!H:!V:VarTransform=Norm:NeuronType=tanh:NCycles=20000:HiddenLayers=" 
    Parameters += self.Layout 
    Parameters += ":TestRate=6:TrainingMethod=BFGS:Sampling=0.3:SamplingEpoch=0.8:ConvergenceImprove=1e-6:ConvergenceTests=15:!UseRegulator"
    Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kMLP, "MLP", Parameters);

    # Train, test, and evaluate internally
    Factory.TrainAllMethods()
    Factory.TestAllMethods()
    Factory.EvaluateAllMethods()



  def test(self):

     
    # (1) Read the data tree
    DataFile = ROOT.TFile(self.FileName);
    if DataFile.IsOpen() == False:
      print("Error: Opening DataFile")
      sys.exit()

    # TODO: Determine string from file name
    DataTree = DataFile.Get("StripPairing_2_2");
    if DataTree == 0:
      print("Error: Reading data tree from root file")
      sys.exit()

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
      
      NumberOfInteractions = int(VariableMap["ResultNumberOfInteractions"][0])
      print("# IAs:      " + str(VariableMap["ResultNumberOfInteractions"][0]) + " vs. " + str(Result[0])) 
      print("Undetected: " + str(VariableMap["ResultUndetectedInteractions"][0]) + " vs. " + str(Result[1])) 
      print("XStripEnergy1: "  + str(VariableMap["XStripEnergy1"][0]) )
      print("XStripEnergy2: "  + str(VariableMap["XStripEnergy2"][0]) )
      print("YStripEnergy1: "  + str(VariableMap["YStripEnergy1"][0]) )
      print("YStripEnergy2: "  + str(VariableMap["YStripEnergy2"][0]) )
    

      ResultInteractions = []
      # Check to see if result interactions are good or bad
      r = 2
      IsSeparable = True
      IsGoodThreshold = 0.3
      for B in list(Branches):
        Name = B.GetName()
        if Name.startswith("ResultInteraction"):
          ResultInteractions.append(VariableMap[Name][0])
          print(Name + str(VariableMap[Name][0]) + " vs. " + str(Result[r]))
          if abs(VariableMap[Name][0] - Result[r]) > IsGoodThreshold:
            IsSeparable = False    
          r += 1

      if IsSeparable == True:
        if NumberOfInteractions == 2:
          NGoodIdentifiedAsGood += 1
          print(" --> Correctly identified separable event")
        else:
          NGoodIdentifiedAsBad += 1
          print(" --> INCORRECTLY identified separable event")          
      else:
        if NumberOfInteractions != 2:
          NBadIdentifiedAsBad += 1
          print(" --> Correctly identified not separable event")
        else:
          NBadIdentifiedAsGood += 1
          print(" --> INCORRECTLY identified not separable event")          
          

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
    print("Good identified as good: " + str(NGoodIdentifiedAsGood)) 
    print("Good identified as bad:  " + str(NGoodIdentifiedAsBad)) 
    print("Bad identified as good:  " + str(NBadIdentifiedAsGood)) 
    print("Bad identified as bad:   " + str(NBadIdentifiedAsBad)) 
    print("Correctly identified events machine learning: " + str(NGoodIdentifiedAsGood + NBadIdentifiedAsBad) + " (" + str(100.0 * (NGoodIdentifiedAsGood + NBadIdentifiedAsBad) / NEvents) + "%)")
    print("Good events test statistic: " + str(NGoodEventsTS) +  " (" + str(100.0 * (NGoodEventsTS) / NEvents) + "%)")

    # prevent Canvases from closing
    #wait()
    print("Close the ROOT window via File -> Close!")
    ROOT.gApplication.Run()


