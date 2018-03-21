import ROOT
import array
import sys


## NEXT STEPS
# upload to github
# split into 2 files
# pass into multiple file_names
# document code
# play with parameters of neural networks (all)
# deeper neural network
# NCycles = 20000
# Hidden Layers: 2*N(hidden layer has twice number of nodes as non-hidden layer)
# TestRate = 6


# more than 100 Nevents

# want to maximize ratio of good events identified as good + bad identified as bad / (good identified as bad + bad identified as good)
# good events defined as EvaluationZenithAngle < 90
# events identified by neural networks


## further down the line
# is EvaluationZenithAngle the best thing to train it on?


### READ file

# filename = "Ling_1000000.seq3.quality.root"


# to run with different seq2 seq3 seq4 files reqplace this line


filename = "Ling.seq3.quality.root"

datafile = ROOT.TFile(filename)

# to run with different seq2 seq3 seq4 files reqplace this line
datatree = datafile.Get("Quality_seq3")
if datatree == 0:
    print("Error reading data tree from root file")
    sys.exit()

### INITIALIZER

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


### FILE INITIALIZER
ignoredbranches = [ 'SimulationID', 'SequenceLength']  #'EvaluationZenithAngle', 'EvaluationIsReconstructable', 'EvaluationIsCompletelyAbsorbed']
branches = datatree.GetListOfBranches()

for name in ignoredbranches:
    dataloader.AddSpectator(name, "F")

for b in list(branches):
    if not b.GetName() in ignoredbranches:
        if not b.GetName().startswith("Evaluation"):
            dataloader.AddVariable(b.GetName(), "F")


# cuts defining the signal and background sample
sigCut = ROOT.TCut("EvaluationZenithAngle > 90") # between 0 and 60 definitely good
bgCut = ROOT.TCut("EvaluationZenithAngle <= 90") # between 60 and 90 maybe good
dataloader.SetInputTrees(datatree, sigCut, bgCut)

dataloader.PrepareTrainingAndTestTree(sigCut,   # signal events
                                      bgCut,    # background events
                                      ":".join([
                                                "nTrain_Signal=0",
                                                "nTrain_Background=0",
                                                "SplitMode=Random",
                                                "NormMode=NumEvents",
                                                "!V"
                                                ]))


# Neural Networks
method = factory.BookMethod(dataloader, ROOT.TMVA.Types.kMLP, "MLP",
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
method = factory.BookMethod(dataloader, ROOT.TMVA.Types.kPDEFoam, "PDEFoamBoost",
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



method = factory.BookMethod(dataloader, ROOT.TMVA.Types.kPDERS, "PDERSPCA",
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

# training0(",".join([
# 	"LearningRate=1e-1",
# 	"Momentum=0.9",
# 	"Repetitions=1",
# 	"ConvergenceSteps=20",
# 	"BatchSize=256",
# 	"TestRepetitions=10",
# 	"WeightDecay=1e-4",
# 	"Regularization=L2",
# 	"DropConfig=0.0+0.5+0.5+0.5",
# 	"Multithreading=True"
# 	]))


factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()

### TESTING

reader = ROOT.TMVA.Reader("!Color:!Silent");
variablemap = {}

for name in ignoredbranches:
    variablemap[name] = array.array('f', [0])
        datatree.SetBranchAddress(name, variablemap[name])
            reader.AddSpectator(name, variablemap[name])

for b in list(branches):
    if not b.GetName() in ignoredbranches:
    if not b.GetName().startswith("Evaluation"):
        variablemap[b.GetName()] = array.array('f', [0])
            reader.AddVariable(b.GetName(), variablemap[b.GetName()])
                datatree.SetBranchAddress(b.GetName(), variablemap[b.GetName()])
                    print("Added: " + b.GetName())

for b in list(branches):
    if b.GetName().startswith("EvaluationZenithAngle") or b.GetName().startswith("EvaluationIsReconstructable") or b.GetName().startswith("EvaluationIsCompletelyAborbed"):
    variablemap[b.GetName()] = array.array('f', [0])
    datatree.SetBranchAddress(b.GetName(), variablemap[b.GetName()])

reader.BookMVA("BDT","Results/weights/TMVAClassification_BDT.weights.xml")

NEvents = 0
NGoodEvents = 0

varx = array.array('f',[0]) #; reader.AddVariable("EvaluationZenithAngle",varx)
vary = array.array('f',[0]) #; reader.AddVariable("result",vary)

for x in range(0, min(100, datatree.GetEntries())):
    datatree.GetEntry(x)
        
        NEvents += 1
            
            print("\nSimulation ID: " + str(int(variablemap["SimulationID"][0])) + ":")
                
                result = reader.EvaluateMVA("BDT")
                    vary.append(result)
                        
                        r = 2
                            IsGood = True
                                IsGoodThreshold = 0.2
                                    for b in list(branches):
                                        name = b.GetName()
                                            if name.startswith("EvaluationZenithAngle"):
                                                print(name + " " + str(variablemap[name][0]) + " vs. " + str(90))
                                                    
                                                    varx.append(variablemap[name][0])
                                                        
                                                        if abs(variablemap[name][0] - 90 > IsGoodThreshold): # and result (type is number) (from neural network) identifies it as good
                                                            IsGood = False
                                                                r += 1
                                                                    elif name.startswith("EvaluationIsReconstructable") or name.startswith("EvaluationIsCompletelyAborbed"):
                                                                        print(name + " " + str(variablemap[name][0]))
                                                                            if not variablemap[name][0]:
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


### Graph Results (y-axis) on EvaluationZenithAngle (x-axis)

# Visualizing the performance:

# keeps objects otherwise removed by garbage collected in a list
gcSaver = []

# create a new TCanvas
gcSaver.append(ROOT.TCanvas())

# create a new 2D histogram with fine binning
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
