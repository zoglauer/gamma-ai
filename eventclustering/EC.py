###################################################################################################
#
# EC.py
#
# Copyright (C) by Andreas Zoglauer.
# All rights reserved.
#
# Please see the file License.txt in the main repository for the copyright-notice. 
#  
###################################################################################################

  
  
###################################################################################################


import ROOT
import array
import os
import sys 
  
  
###################################################################################################


class EventClustering:
  """
  This class performs event clustering training. A typical usage would look like this:

  AI = EventClustering("EC.maxhits3.eventclusterizer.root", False, "Results", "3*N,N", "MLP", "100000")
  AI.train()
  AI.test()

  """

  
###################################################################################################


  def __init__(self, FileName, OutputPrefix, Algorithms, NetworkLayout, EnergyBins, MaxEvents):
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
    NetworkLayout: string
      The layout of the neural network (e.g. "3*N,N")
    MaxEvents: integer
      The maximum amount of events to use
    
    """ 
    
    self.FileName = FileName
    self.OutputPrefix = OutputPrefix
    self.Algorithms = Algorithms.split(",")
    self.NetworkLayout = NetworkLayout
    self.EnergyBins = [int(E) for E in EnergyBins.split(",")]
    self.MaxEvents = MaxEvents
    
    if len(self.EnergyBins) < 2:
      print("ERROR: You need at least 2 energy bins. Using [0, 10000]")
      self.EnergyBins = [ 0, 10000 ]
 
    if os.sep in self.OutputPrefix:
      print("ERROR: The output prefix is just a name not file path, thus it cannot contain any file seperators. Using \"Results\" as the prefix.")
      self.OutputPrefix = "Results"
 
  
###################################################################################################


  def train(self, TrainAll):
    """
    Main training function - splits between training just the file named in the constructor (TrainAll == False)
    or training all similar files it finds, e.g. 
    - X.maxhits2.eventclusterizer.root 
    - X.maxhits3.eventclusterizer.root 
    - X.maxhits4.eventclusterizer.root 
    etc.
    
    Attributes
    ----------
    TrainAll : bool
      Indicates if all similar files hsould be trained (== True), or just the one given in the
      constructor (== False)
    
    Returns
    -------
    bool
      True is everything went well, False in case of an error 
      
    """
    
    
    # Check if we should go through multiple or just one file name
    if TrainAll == True:
      # Find all other files
      
      FileNames = self.findDataSetFiles(self.FileName)

      if len(FileNames) == 0:
        print("ERROR: No usable data files found!")
        return False

      for Name in FileNames:
        for e in range(1, len(self.EnergyBins)):
          self.trainIndividual(Name, self.EnergyBins[e-1], self.EnergyBins[e])
      
    else:
      for e in range(1, len(self.EnergyBins)):
        self.trainIndividual(self.FileName, self.EnergyBins[e-1], self.EnergyBins[e])
    
    return True
  
  
###################################################################################################


  def trainIndividual(self, FileName, MinimumEnergy, MaximumEnergy):
    """
    Train on the given file
    
    Attributes
    ----------
    FileName : string
      The file name of the data set used for training
    
    Returns
    -------
    bool
      True is everything went well, False in case of an error 
      
    """
    
    # Extract the number of hits
    (NumberOfHits, NumberOfGroups) = self.getNumberOfHitsAndGroups(FileName)
    if NumberOfHits == 0:
      print("Error: Unable to extract the hit multiplicity")
      return False
    if NumberOfHits == 1:
      print("Error: You need at least 2 hits for training. The data set is one with only 1 hit per event")
      return False
      
    # Read the data tree
    DataFile = ROOT.TFile(FileName);
    if DataFile.IsOpen() == False:
      print("Error: Opening DataFile")
      return False

    # Extract the data tree
    FullDataTree = DataFile.Get("EventClusterizer");
    if FullDataTree == 0:
      print("Error: Reading data tree from root file")
      return False

    # Filter energy
    ROOT.gROOT.cd()
    EnergyString = "Energy_1"
    for i in range(1, NumberOfHits):
      EnergyString += " + Energy_" + str(i+1)
    
    DataTree = FullDataTree.CopyTree("(" + EnergyString + ") >= " + str(MinimumEnergy) + " && (" + EnergyString + ") <= " + str(MaximumEnergy), "")
    
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
    if not os.path.exists(self.OutputPrefix):
      os.makedirs(self.OutputPrefix)
    
    FullPrefix = self.OutputPrefix + os.sep + self.OutputPrefix + ".hits" + str(NumberOfHits) + ".emin" + str(MinimumEnergy) + ".emax" + str(MaximumEnergy)
    
    ResultsFile = ROOT.TFile(FullPrefix + ".root", "RECREATE")


    # Create the Factory, responsible for training and evaluation
    Factory = ROOT.TMVA.Factory("TMVARegression", ResultsFile, "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Regression")

    # Create the data loader - give it the name of the output directory
    DataLoader = ROOT.TMVA.DataLoader(FullPrefix)

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
    for B in list(Branches):
      if B.GetName().startswith("Result"):
        DataLoader.AddTarget(B.GetName(), "F")


    # Add the regressions tree with weight = 1.0
    DataLoader.AddRegressionTree(DataTree, 1.0);


    # Random split between training and test data
    Cut = ROOT.TCut("")
    DataLoader.PrepareTrainingAndTestTree(Cut, "SplitMode=Random:SplitSeed=0:V");

    print(self.Algorithms)

    # Book a multi-layer perceptron
    if 'MLP' in self.Algorithms:
      Parameters = ROOT.TString()
      Parameters += "!H:!V:VarTransform=Norm:NeuronType=tanh:NCycles=20000:HiddenLayers=" 
      Parameters += self.NetworkLayout 
      Parameters += ":TestRate=6:TrainingMethod=BFGS:Sampling=0.3:SamplingEpoch=0.8:ConvergenceImprove=1e-6:ConvergenceTests=15:!UseRegulator"
      Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kMLP, "MLP", Parameters);

    # Book the DNN approach:
    if 'DNNCPU' in self.Algorithms:
      Parameters = ROOT.TString()
      Parameters += "!H:!V:VarTransform=Norm:NeuronType=tanh:NCycles=20000:HiddenLayers=" 
      Parameters += self.NetworkLayout 
      Parameters += ":TestRate=6:TrainingMethod=BFGS:Sampling=0.3:SamplingEpoch=0.8:ConvergenceImprove=1e-6:ConvergenceTests=15:!UseRegulator"
      Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kMLP, "MLP", Parameters);


    # Train, test, and evaluate internally
    Factory.TrainAllMethods()
    Factory.TestAllMethods()
    Factory.EvaluateAllMethods()
    
    return True
  
  
###################################################################################################


  def test(self, TestAll):
    """
    Main test function - splits between testing just the file named in the constructor (TrainAll == False)
    or testing all similar files it finds, e.g. 
    - X.maxhits2.eventclusterizer.root 
    - X.maxhits3.eventclusterizer.root 
    - X.maxhits4.eventclusterizer.root 
    etc.
    
    Attributes
    ----------
    TestAll : bool
      Indicates if all similar files hsould be trained (== True), or just the one given in the
      constructor (== False)
    
    Returns
    -------
    bool
      True is everything went well, False in case of an error 
      
    """
    
    # Check if we should go through multiple or just one file name
    if TestAll == True:
      # Find all other files
      
      FileNames = self.findDataSetFiles(self.FileName)

      if len(FileNames) == 0:
        print("ERROR: No usable data files found!")
        return False

      for Name in FileNames:
        for e in range(1, len(self.EnergyBins)):
          self.testIndividual(Name, self.EnergyBins[e-1], self.EnergyBins[e])
      
    else:
      for e in range(1, len(self.EnergyBins)):
        self.testIndividual(self.FileName, self.EnergyBins[e-1], self.EnergyBins[e])
    
    return True
  
  
###################################################################################################


  def testIndividual(self, FileName, MinimumEnergy, MaximumEnergy):
    """
    Test the given file
    
    Attributes
    ----------
    FileName : string
      The file name of the data set used for training
    
    Returns
    -------
    bool
      True is everything went well, False in case of an error 
      
    """
  
    # Extract the number of hits
    (NumberOfHits, NumberOfGroups) = self.getNumberOfHitsAndGroups(FileName)
    if NumberOfHits == 0:
      print("Error: Unable to extract the hit multiplicity")
      return False
    if NumberOfHits == 1:
      print("Error: You need at least 2 hits for training. The data set is one with only 1 hit per event")
      return False  
  
    # Open the data set
    DataFile = ROOT.TFile(FileName);
    if DataFile.IsOpen() == False:
      print("Error: Opening data file")
      return False

    # Extract the data tree
    FullDataTree = DataFile.Get("EventClusterizer");
    if FullDataTree == 0:
      print("Error: Reading data tree from root file")
      return False
    
    # Filter energy
    ROOT.gROOT.cd()
    EnergyString = "Energy_1"
    for i in range(1, NumberOfHits):
      EnergyString += " + Energy_" + str(i+1)
    
    DataTree = FullDataTree.CopyTree("(" + EnergyString + ") >= " + str(MinimumEnergy) + " && (" + EnergyString + ") <= " + str(MaximumEnergy), "")
    
    # Limit the number of events:
    if DataTree.GetEntries() > self.MaxEvents:
      print("Reducing source tree size from " + str(DataTree.GetEntries()) + " to " + str(self.MaxEvents) + " (i.e. the maximum set)")
      NewTree = DataTree.CloneTree(0);
      NewTree.SetDirectory(0);
    
      for i in range(0, self.MaxEvents):
        DataTree.GetEntry(i)
        NewTree.Fill()
    
      DataTree = NewTree
      
      

    # Initialize TMVA
    ROOT.TMVA.Tools.Instance()
     
    IgnoredBranches = [ 'SimulationID' ]  
    Branches = DataTree.GetListOfBranches()

    # Setup the reader:
    Reader = ROOT.TMVA.Reader("!Color:Silent");    

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

    # Add the target variables:
    for B in list(Branches):
      if B.GetName().startswith("Result"):
        VariableMap[B.GetName()] = array.array('f', [0])
        DataTree.SetBranchAddress(B.GetName(), VariableMap[B.GetName()])
        
        
    FileName = ROOT.TString(self.OutputPrefix + os.sep + self.OutputPrefix + ".hits" + str(NumberOfHits) + ".emin" + str(MinimumEnergy) + ".emax" + str(MaximumEnergy)
 + "/weights/TMVARegression_MLP.weights.xml")
    Reader.BookMVA("MLP", FileName)

    # Intialize statistics
    NEvents = 0
    NGood = 0
    NBad = 0
    
    # Create histograms of the test statistic values:
        
    # keeps objects otherwise removed by garbage collected in a list
    ROOTSaver = []



    # Read simulated the events
    for x in range(0, min(self.MaxEvents, DataTree.GetEntries())):
      DataTree.GetEntry(x)
      
      NEvents += 1
      
      # First extract the input
      TrainingResults = []
      for B in list(Branches):
        Name = B.GetName()
        if Name.startswith("ResultHitGroups"):
          TrainingResults.append(VariableMap[Name][0])
      
      # Do the evaluation
      MLResults = Reader.EvaluateRegression("MLP")  
      
      # Compare Training and ML results
      Agree = True
      for t, m in zip(TrainingResults, MLResults):
        rt = round(abs(t), 0)
        rm = round(abs(m), 0)
        if rt != rm:
          Agree = False
              
      #print("\nSimulation ID: " + str(int(VariableMap["SimulationID"][0])) + ":")
      #print("Energies: " + str(VariableMap["Energy_1"][0]) + " " + str(VariableMap["Energy_2"][0]) + " " + str(VariableMap["Energy_3"][0]))
      #for t, m in zip(TrainingResults, MLResults):
      #  print("%.1f vs %.1f" % (round(abs(t), 1), round(abs(m), 1)))
      
      if Agree == True:
        NGood += 1
        #print("---> Good")
      else:
        NBad += 1
        #print("---> Bad")
        
    # Dump some statistics:
    print("\n\n")
    print("Number of hits: " + str(NumberOfHits) + "  -- energy range: " + str(MinimumEnergy) + "-" + str(MaximumEnergy))
    print("All events: " + str(NEvents))
    print("Good: " + str(100*NGood/NEvents) + "%") 
    print("Bad: " + str(100*NBad/NEvents) + "%")
    print("\n")
    
  
    return True
  
  
###################################################################################################


  def getNumberOfHitsAndGroups(self, FileName):
    """
    Return the number of hits and groups from the file name
    
    Attributes
    ----------
    FileName : string
      The file name of the data set used for training
    
    Returns
    -------
    (int, int)
      The number of hits and the number of groups found, 0 in case of error
        
    """
    
    NumberOfHits = 0
    NumberOfGroups = 0
    
    Split = FileName.split(".hits");
    if len(Split) != 2:
      print("ERROR: Unable to find the hit multiplicity in file " + FileName)
      return (0, 0)
    
    Split = Split[1].split(".groups");
    if len(Split) != 2:
      print("ERROR: Unable to find the hit multiplicity in file " + FileName)
      return (0, 0)
    
    NumberOfHits = int(Split[0])

    Split = Split[1].split(".eventclusterizer.root");
    if len(Split) != 2:
      print("ERROR: Unable to find the group multiplicity in file " + FileName)
      return (0, 0)
    
    NumberOfGroups = int(Split[0])
    
    return (NumberOfHits, NumberOfGroups) 
  
  
###################################################################################################


  def findDataSetFiles(self, FileName):
    """
    Find and return all data set given one file name
    
    Attributes
    ----------
    FileName : string
      The file name of the data set used for training
    
    Returns
    -------
    [] of strings
      A list of the file names, empty when none were found
        
    """
    
    FileNames = []

    # Find the file prefix - everything before max hits 
    Split = FileName.split(".hits");
    if len(Split) != 2:
      print("ERROR: Unable to find file prefix for file " + FileName)
      return FileNames
      
    Prefix = Split[0]
    
    (NumberOfHits, NumberOfGroups) = self.getNumberOfHitsAndGroups(FileName)

    for s in range(2, 1000):
      Name = Prefix + ".hits" + str(s) + ".groups" + str(NumberOfGroups) + ".eventclusterizer.root";

      if os.path.isfile(Name) == True:
        FileNames.append(Name)
      else:
        break
        
    return FileNames


# END  
###################################################################################################
