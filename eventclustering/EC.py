
import ROOT
import array
import sys 


# TODO: Seperate common loading

class EventClustering:

  def __init__(self, FileName, OutputPrefix, Layout, Algorithms, MaxEvents):
    self.FileName = FileName
    self.Layout = Layout
    self.OutputPrefix = OutputPrefix
    self.Algorithms = Algorithms.split(",")
    self.MaxEvents = int(MaxEvents)
    
    

  def train(self):
     
    # Read the data tree
    DataFile = ROOT.TFile(self.FileName);
    if DataFile.IsOpen() == False:
      print("Error: Opening DataFile")
      sys.exit()

    # Extract the data tree
    DataTree = DataFile.Get("EventClusterizer");
    if DataTree == 0:
      print("Error: Reading data tree from root file")
      sys.exit()


    TreeSize = DataTree.GetEntries();
  
    if TreeSize > self.MaxEvents:
      print("Reducing source tree size from " + str(TreeSize) + " to " + str(self.MaxEvents) + " (i.e. the maximum set)")
      NewTree = DataTree.CloneTree(0);
      NewTree.SetDirectory(0);
    
      for i in range(0, self.MaxEvents):
        DataTree.GetEntry(i)
        NewTree.Fill()
    
      DataTree = NewTree;


    # Initialize TMVA
    ROOT.TMVA.Tools.Instance()
     

     
    # PART 1: Train the neural network 
     
     
    # The output file
    ResultsFileName = self.OutputPrefix + ".root"
    ResultsFile = ROOT.TFile(ResultsFileName, "RECREATE")


    # Create the Factory, responsible for training and evaluation
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
    for B in list(Branches):
      if B.GetName().startswith("Result"):
        DataLoader.AddTarget(B.GetName(), "F")


    # Add the regressions tree with weight = 1.0
    DataLoader.AddRegressionTree(DataTree, 1.0);


    # Random split between training and test data
    Cut = ROOT.TCut("")
    DataLoader.PrepareTrainingAndTestTree(Cut, "SplitMode=Random:SplitSeed=0:V");


    # Book a multi-layer perceptron
    if 'MLP' in self.Algorithms:
      Parameters = ROOT.TString()
      Parameters += "!H:!V:VarTransform=Norm:NeuronType=tanh:NCycles=20000:HiddenLayers=" 
      Parameters += self.Layout 
      Parameters += ":TestRate=6:TrainingMethod=BFGS:Sampling=0.3:SamplingEpoch=0.8:ConvergenceImprove=1e-6:ConvergenceTests=15:!UseRegulator"
      Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kMLP, "MLP", Parameters);

    # Book the DNN approach:
    if 'DNNCPU' in self.Algorithms:
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

    DataTree = DataFile.Get("EventClusterizer");
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
    for B in list(Branches):
      if B.GetName().startswith("Result"):
        VariableMap[B.GetName()] = array.array('f', [0])
        DataTree.SetBranchAddress(B.GetName(), VariableMap[B.GetName()])
        
        


    FileName = ROOT.TString(self.OutputPrefix)
    FileName += "/weights/TMVARegression_MLP.weights.xml"
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
              
      print("\nSimulation ID: " + str(int(VariableMap["SimulationID"][0])) + ":")
      print("Energies: " + str(VariableMap["Energy_1"][0]) + " " + str(VariableMap["Energy_2"][0]) + " " + str(VariableMap["Energy_3"][0]))
      for t, m in zip(TrainingResults, MLResults):
        print("%.1f vs %.1f" % (round(abs(t), 1), round(abs(m), 1)))
      
      if Agree == True:
        NGood += 1
        print("---> Good")
      else:
        NBad += 1
        print("---> Bad")
        
    # Dump some statistics:
    print("All events: " + str(NEvents))
    print("Good: " + str(NGood)) 
    print("Bad: " + str(NBad)) 
    
    

    
    # create a new TCanvas





