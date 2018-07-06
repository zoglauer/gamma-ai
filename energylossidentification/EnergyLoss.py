###################################################################################################
#
# EnergyLoss.py
#
# Copyright (C) by Andreas Zoglauer.
# All rights reserved.
#
# Please see the file License.txt in the main repository for the copyright-notice. 
#  
###################################################################################################


# TODO: Train and test all multiplicities
# TODO: Test performance as a function of energy
# TODO: Test performance as a function of zenith angle
# TODO: Test deep neural Networks
# TODO: Test different libraries

  
###################################################################################################


import ROOT
import array
import os
import sys 

  
###################################################################################################


class EnergyLossIdentification:
  """
  This class performs energy loss training. A typical usage would look like this:

  AI = EnergyLossIdentification("Ling2.seq3.quality.root", "Results", "MLP,BDT", 1000000)
  AI.train()
  AI.test()

  """

  
###################################################################################################

  
  def __init__(self, FileName, Output, Algorithm, MaxEvents):
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
    MaxEvents: integer
      The maximum amount of events to use
    
    """ 
    
    self.FileName = FileName
    self.OutputPrefix = Output
    self.Algorithms = Algorithm
    self.MaxEvents = MaxEvents

  
###################################################################################################


  def train_scikitlearn(self):

    # Open the file
    DataFile = ROOT.TFile(self.FileName)
    if DataFile.IsOpen() == False:
      print("Error opening data file")
      return False

    # Get the data tree
    DataTree = DataFile.Get("Quality")
    if DataTree == 0:
      print("Error reading data tree from root file")
      return False

    Branches = DataTree.GetListOfBranches()

    VariableMap = {}

    # Create a map of the branches, i.e. the columns
    for B in list(Branches):
      if B.GetName() == "EvaluationIsCompletelyAbsorbed":
        VariableMap[B.GetName()] = array.array('i', [0]) # why put f here?
      else:
        VariableMap[B.GetName()] = array.array('f', [0]) # why put f here?
      DataTree.SetBranchAddress(B.GetName(), VariableMap[B.GetName()])
      #print("Adding branch: " + B.GetName())

    # Read simulated the events
    import numpy as np

    X_data=np.zeros((1,40)) # space holder
    y_data=np.zeros((1))

    all_features=VariableMap.keys()
    all_features.remove("SequenceLength")
    all_features.remove("SimulationID")
    all_features.remove("EvaluationIsReconstructable")
    all_features.remove("EvaluationZenithAngle")

    all_features.remove("EvaluationIsCompletelyAbsorbed") #y
    



    #print("total rows", DataTree.GetEntries())
    #for x in range(0, 5000):
    for x in range(0, min(10000, DataTree.GetEntries())):
      if x%1000 == 0 and x > 0:
        print("Progress: " + str(x) + "/" + str(DataTree.GetEntries()))

      DataTree.GetEntry(x)  # Get row x
      #DataTree.Show()

      #print("abs: " + str(VariableMap["EvaluationIsCompletelyAbsorbed"][0]))
      #print("energy1: " + str(VariableMap["Energy1"][0])) 

      new_row=[VariableMap[feature][0] for feature in all_features]
      #print("new_row: ",new_row)
      X_data=np.vstack((X_data, np.array(new_row)))

      # cut=0.5
      
      if VariableMap["EvaluationIsCompletelyAbsorbed"][0] == 1:
        target=1.0
        #print("1")
      else:
        target=0.0
        #print("0")
      y_data=np.append(y_data, [target])
    
    # remove place holder

    X_data = np.delete(X_data, (0), axis=0)
    y_data = np.delete(y_data, 0)
    #print("total X: ", len(X_data))
    #print("total y: ", len(y_data))
    """
    /Users/winnielee/code/.virtualenvs/cosi2.7/lib/python2.7/site-packages/sklearn/metrics/classification.py:1428: UserWarning: labels size, 1, does not match size of target_names, 2
      .format(len(labels), len(target_names))
    """
    #print(X_data, y_data)
    

    # alternative: use root_numpy to get data from Root Tree

    """from root_numpy import root2array, rec2array
    branch_names = VariableMap.keys()

    signal = root2array(DataTree, "tree", branch_names)
    signal = rec2array(signal)

    # ?
    backgr = root2array("", "tree", branch_names)
    backgr = rec2array(backgr)

    # create 2d numpy array for scikit-learn
    X_data = np.concatenate((signal, backgr))
    y_data = np.concatenate((np.ones(signal.shape[0]), np.zeros(backgr.shape[0])))"""

    
    #############scikit-learn
    
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    
    print("start")
    # split train-test data
    X_train,X_test, y_train,y_test = train_test_split(X_data, y_data, test_size=0.5, random_state=0)
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.05)
    bdt = AdaBoostClassifier(dt,
                         algorithm='SAMME',
                         n_estimators=800,
                         learning_rate=0.1)

    # train
    bdt.fit(X_train, y_train)
    # test
    y_predicted = bdt.predict(X_test)
    #print("X_test",len(X_test))
    # evaluate (roc curve)
    print classification_report(y_test, y_predicted, target_names=["background", "signal"])
    print "Area under ROC curve: %.4f"%(roc_auc_score(y_test, y_predicted))
      #other evaluation

    # parameter adjustments
    # - learning rate
    # - scaling? energy value is larger but only around 1k~10k times

###################################################################################################


  def train(self):
    """
    Main training function 
    
    Returns
    -------
    bool
      True is everything went well, False in case of an error 
      
    """
    # Open the file
    DataFile = ROOT.TFile(self.FileName)
    if DataFile.IsOpen() == False:
      print("Error opening data file")
      return False

    # Get the data tree
    DataTree = DataFile.Get("Quality")
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
    ResultsFile = ROOT.TFile(FullPrefix + ".root", "RECREATE")

    Factory = ROOT.TMVA.Factory("TMVAClassification", ResultsFile, "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification")

    DataLoader = ROOT.TMVA.DataLoader(self.OutputPrefix)

    IgnoredBranches = [ 'SimulationID', 'SequenceLength']
    Branches = DataTree.GetListOfBranches()

    for Name in IgnoredBranches:
      DataLoader.AddSpectator(Name, "F")

    for B in list(Branches):
      if not B.GetName() in IgnoredBranches:
        if not B.GetName().startswith("Evaluation"):
          DataLoader.AddVariable(B.GetName(), "F")

    SignalCut = ROOT.TCut("EvaluationIsCompletelyAbsorbed >= 0.5")
    BackgroundCut = ROOT.TCut("EvaluationIsCompletelyAbsorbed < 0.5")
    DataLoader.SetInputTrees(DataTree, SignalCut, BackgroundCut)

    DataLoader.PrepareTrainingAndTestTree(SignalCut, BackgroundCut, "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V")
    print(DataLoader)
    return False
    # Neural Networks
    if 'MLP' in self.Algorithms:
      method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kMLP, "MLP", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=100:HiddenLayers=N+10,N-5:TestRate=5:TrainingMethod=BFGS:!UseRegulator")
      #method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kMLP, "MLP", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=100:HiddenLayers=N+10,N-5:TestRate=5:!UseRegulator")
      #method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kMLP, "MLP", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=100:HiddenLayers=N+10,N-5:TestRate=6:TrainingMethod=BFGS:Sampling=0.3:SamplingEpoch=0.8:ConvergenceImprove=1e-6:ConvergenceTests=15:!UseRegulator")


    # PDEFoamBoost
    if 'PDEFoamBoost' in self.Algorithms:
      method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kPDEFoam, "PDEFoamBoost", "!H:!V:Boost_Num=100:Boost_Transform=linear:SigBgSeparate=F:MaxDepth=4:UseYesNoCell=T:DTLogic=MisClassificationError:FillFoamWithOrigWeights=F:TailCut=0:nActiveCells=2000:nBin=50:Nmin=200:Kernel=None:Compress=T")
      #method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kPDEFoam, "PDEFoamBoost", "!H:!V:Boost_Num=30:Boost_Transform=linear:SigBgSeparate=F:MaxDepth=4:UseYesNoCell=T:DTLogic=MisClassificationError:FillFoamWithOrigWeights=F:TailCut=0:nActiveCells=500:nBin=20:Nmin=400:Kernel=None:Compress=T")

    # PDERSPCA
    if 'PDERSPCA' in self.Algorithms:
      method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kPDERS, "PDERSPCA", "!H:!V:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600:VarTransform=PCA")

    # Random Forest Boosted Decision Trees
    if 'BDT' in self.Algorithms:
      method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kBDT, "BDT", "!H:!V:NTrees=1000:MinNodeSize=1%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.4:SeparationType=CrossEntropy:nCuts=100:PruneMethod=NoPruning")
      #method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kBDT, "BDT", "!H:!V:NTrees=850:nEventsMin=150:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning")
      #method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kBDT, "BDT", "!H:!V:NTrees=1000:nEventsMin=1000:MaxDepth=4:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning")

    # State Vector Machine
    if 'SVM' in self.Algorithms:
      method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kSVM, "SVM", "Gamma=0.25:Tol=0.001:VarTransform=Norm");

    # DNN
    if 'DNN_CPU' in self.Algorithms:
      Layout = "Layout=TANH|N,TANH|N/2,LINEAR"

      Training0 = "LearningRate=1e-1,Momentum=0.9,Repetitions=1,ConvergenceSteps=30,BatchSize=256,TestRepetitions=10,WeightDecay=1e-4,Regularization=L2,DropConfig=0.0+0.5+0.5+0.5,Multithreading=True"
      Training1 = "LearningRate=1e-2,Momentum=0.9,Repetitions=1,ConvergenceSteps=30,BatchSize=256,TestRepetitions=10,WeightDecay=1e-4,Regularization=L2,DropConfig=0.0+0.0+0.0+0.0,Multithreading=True"
      Training2 = "LearningRate=1e-3,Momentum=0.0,Repetitions=1,ConvergenceSteps=30,BatchSize=256,TestRepetitions=10,WeightDecay=1e-4,Regularization=L2,DropConfig=0.0+0.0+0.0+0.0,Multithreading=True"
      TrainingStrategy = "TrainingStrategy=" + Training0 + "|" + Training1 + "|" + Training2

      Options = "!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:WeightInitialization=XAVIERUNIFORM:" + Layout + ":" + TrainingStrategy
      
      Options += ":Architecture=CPU"
      Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kDNN, "DNN_CPU", Options)


    # DNN
    if 'DNN_GPU' in self.Algorithms:
      Layout = "Layout=TANH|N,TANH|N/2,LINEAR"

      Training0 = "LearningRate=1e-1,Momentum=0.9,Repetitions=1,ConvergenceSteps=100,BatchSize=256,TestRepetitions=10,WeightDecay=1e-4,Regularization=L2,DropConfig=0.0+0.5+0.5+0.5,Multithreading=True"
      Training1 = "LearningRate=1e-2,Momentum=0.9,Repetitions=1,ConvergenceSteps=100,BatchSize=256,TestRepetitions=10,WeightDecay=1e-4,Regularization=L2,DropConfig=0.0+0.0+0.0+0.0,Multithreading=True"
      Training2 = "LearningRate=1e-3,Momentum=0.0,Repetitions=1,ConvergenceSteps=100,BatchSize=256,TestRepetitions=10,WeightDecay=1e-4,Regularization=L2,DropConfig=0.0+0.0+0.0+0.0,Multithreading=True"
      TrainingStrategy = "TrainingStrategy=" + Training0 + "|" + Training1 + "|" + Training2

      Options = "!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:WeightInitialization=XAVIERUNIFORM:" + Layout + ":" + TrainingStrategy
      
      Options += ":Architecture=GPU"
      Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kDNN, "DNN_GPU", Options)


    # Finally test, train & evaluate all methods
    Factory.TrainAllMethods()
    Factory.TestAllMethods()
    Factory.EvaluateAllMethods()

    return True

  
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
