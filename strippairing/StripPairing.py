# Machine learning for strip pairing

import ROOT
import array
import sys 
 
#FileName = "StripPairing.2MeV.x2.y2.strippairing.root"
FileName = "StripPairing.x2.y2.strippairing.root"
 
# (1) Read the data tree

DataFile = ROOT.TFile(FileName);

DataTree = DataFile.Get("StripPairing_2_2");
if DataTree == 0:
  print("Error reading data tree from root file")
  sys.exit()

# Initialize TMVA
ROOT.TMVA.Tools.Instance()
 

 
# PART 1: Train the neural networl 
 
 
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
Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kMLP, "MLP", "!H:!V:VarTransform=Norm:NeuronType=tanh:NCycles=20000:HiddenLayers=N+20:TestRate=6:TrainingMethod=BFGS:Sampling=0.3:SamplingEpoch=0.8:ConvergenceImprove=1e-6:ConvergenceTests=15:!UseRegulator" );

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


NEvents = 0
NGoodEvents = 0

for x in range(0, min(100, DataTree.GetEntries())):
  DataTree.GetEntry(x)
  
  NEvents += 1
  
  print("\nSimulation ID: " + str(int(VariableMap["SimulationID"][0])) + ":")
  
  #print("Input variables:")
  #print(VariableMap)
  
  Result = Reader.EvaluateRegression("MLP")  
  #print("Result:" )
  #for R in Result:
  #  print(R)
    
  print("# IAs:      " + str(VariableMap["ResultNumberOfInteractions"][0]) + " vs. " + str(Result[0])) 
  print("Undetected: " + str(VariableMap["ResultUndetectedInteractions"][0]) + " vs. " + str(Result[1])) 

  r = 2
  IsGood = True
  IsGoodThreshold = 0.2
  for B in list(Branches):
    Name = B.GetName()
    if Name.startswith("ResultInteraction"):
      print(Name + str(VariableMap[Name][0]) + " vs. " + str(Result[r]))
      if abs(VariableMap[Name][0] - Result[r]) > IsGoodThreshold:
        IsGood = False    
      r += 1

  if IsGood == True:
    NGoodEvents += 1
    print(" --> Good event")
  else:
    print(" --> Bad event")

print("\nResult:")
print("All events: " + str(NEvents))
print("Good events: " + str(NGoodEvents))


#ROOT.TMVA.TMVARegGui(ResultsFileName);


# prevent Canvases from closing
#print("Close the ROOT window via File -> Close!")
#ROOT.gApplication.Run()


