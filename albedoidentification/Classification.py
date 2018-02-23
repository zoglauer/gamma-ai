# Based on original example by Andre Holzner published under Apache License Version 2.0
# Web: https://aholzner.wordpress.com/2011/08/27/a-tmva-example-in-pyroot/

import ROOT
import array
import sys 

FileName = "Classification.x2.y2.classification.root"
 
# (1) Read the data tree

DataFile = ROOT.TFile(FileName);

DataTree = DataFile.Get("Classification_2_2");
if DataTree == 0:
  print("Error reading data tree from root file")
  sys.exit()

# Initialize TMVA
ROOT.TMVA.Tools.Instance()

 
# note that it seems to be mandatory to have an
# output file, just passing None to TMVA::Factory(..)
# does not work. Make sure you don't overwrite an
# existing file.
fout = ROOT.TFile("Results.root","RECREATE")
 
factory = ROOT.TMVA.Factory("TMVAClassification", fout,
                            ":".join([
                                "!V",
                                "!Silent",
                                "Color",
                                "DrawProgressBar",
                                "Transformations=I;D;P;G,D",
                                "AnalysisType=Classification"]
                                     ))

dataloader = ROOT.TMVA.DataLoader("Results")

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
 
# cuts defining the signal and background sample
sigCut = ROOT.TCut("signal > 0.5")
bgCut = ROOT.TCut("signal <= 0.5")
 
dataloader.PrepareTrainingAndTestTree(sigCut,   # signal events
                                   bgCut,    # background events
                                   ":".join([
                                        "nTrain_Signal=0",
                                        "nTrain_Background=0",
                                        "SplitMode=Random",
                                        "NormMode=NumEvents",
                                        "!V"
                                       ]))

method = factory.BookMethod(dataloader, ROOT.TMVA.Types.kBDT, "BDT",
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
 
 
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()


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

  
  Result = Reader.EvaluateRegression("MLP")  

    
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


# prevent Canvases from closing
print("Close the ROOT window via File -> Close!")
ROOT.gApplication.Run()


