
import ROOT
import array
import sys 
 

import itertools
import permutations

class StripPairing:

  def __init__(self, filename, variable):
    self.filename = filename
    self.variable = variable

  def run(self):
#put multiples and permutations in a separate file
#upload to github
#play with neural network settings 

    FileName = self.filename
     
    # (1) Read the data tree

    DataFile = ROOT.TFile(FileName);
    if DataFile.IsOpen() == False:
      print("Error opening DataFile")
      sys.exit()

    DataTree = DataFile.Get("StripPairing_2_2");
    if DataTree == 0:
      print("Error reading data tree from root file")
      sys.exit()

    # Initialize TMVA
    ROOT.TMVA.Tools.Instance()
     

     
    # PART 1: Train the neural network 
     
     
    # The output file
    ResultsFileName = "Results.root"
    ResultsFile = ROOT.TFile(ResultsFileName, "RECREATE")

    # Create the Factory, responible for training and evaluation
    Factory = ROOT.TMVA.Factory("TMVARegression", ResultsFile, "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Regression")

    # Create the data loader - give it the name of the output directory
    DataLoader = ROOT.TMVA.DataLoader("Results")

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
    Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kMLP, "MLP", "!H:!V:VarTransform=Norm:NeuronType=tanh:NCycles=20000:HiddenLayers="+self.variable+":TestRate=6:TrainingMethod=BFGS:Sampling=0.3:SamplingEpoch=0.8:ConvergenceImprove=1e-6:ConvergenceTests=15:!UseRegulator" );

    # Train, test, and evaluate internally
    Factory.TrainAllMethods()
    Factory.TestAllMethods()
    Factory.EvaluateAllMethods()





    # PART 2: Testing what we just trained

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


    Reader.BookMVA("MLP","Results/weights/TMVARegression_MLP.weights.xml")

    #intialize events
    NEvents = 0
    NGoodEvents = 0
    NGoodEventsTS = 0
    #simulate the events
    for x in range(0, min(100, DataTree.GetEntries())):
      DataTree.GetEntry(x)
      
      NEvents += 1
      
      print("\nSimulation ID: " + str(int(VariableMap["SimulationID"][0])) + ":")
      
      Result = Reader.EvaluateRegression("MLP")  
        
      print("# IAs:      " + str(VariableMap["ResultNumberOfInteractions"][0]) + " vs. " + str(Result[0])) 
      print("Undetected: " + str(VariableMap["ResultUndetectedInteractions"][0]) + " vs. " + str(Result[1])) 

      ResultInteractions = []
      #Check to see if result interactions are good or bad
      r = 2
      IsGood = True
      IsGoodThreshold = 0.2
      for B in list(Branches):
        Name = B.GetName()
        if Name.startswith("ResultInteraction"):
          ResultInteractions.append(VariableMap[Name][0])
          print(Name + str(VariableMap[Name][0]) + " vs. " + str(Result[r]))
          if abs(VariableMap[Name][0] - Result[r]) > IsGoodThreshold:
            IsGood = False    
          r += 1

      if IsGood == True:
        NGoodEvents += 1
        print(" --> Good event")
      else:
        print(" --> Bad event")

      # make list of X and Y strip energies
      XStripList = []
      YStripList = []
      for B in Branches:
        if B.GetName().startswith("XStripEnergy"):
          XStripList.append(VariableMap[B.GetName()][0])
        elif B.GetName().startswith("YStripEnergy"):
          YStripList.append(VariableMap[B.GetName()][0])

      #generate comparison array fromm ResultsInteraction
      idealinteractions = []
      for B in Branches:
        if B.GetName().startswith("ResultInteraction"):
          idealinteractions.append(VariableMap[B.GetName()][0])

      NX = len(XStripList)
      NY = len(YStripList)

      Result = permutations.CreateStripCombinations(NX, NY)
            
      print(Result)

    #make the test statistic
      Ts = []
      for grouping in Result:
        N= (len(grouping))
        total = 0
        for pair in grouping:
          x = pair[0]
          y = pair[1]
          total = total + (XStripList[x]-YStripList[y])**2
        Ts.append(total * (1/N))
    #find the minimum from the test statistic
      import numpy as np
      index_min = np.argmin(Ts)

      RITest = np.zeros(len(ResultInteractions)) 
    #if it is correct, change it to a 1
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
        print("Good event from TS!")




    print("\nResult:")
    print("All events: " + str(NEvents))
    print("Good events machine learning: " + str(NGoodEvents))
    print("Good events test statistic: " + str(NGoodEventsTS))




