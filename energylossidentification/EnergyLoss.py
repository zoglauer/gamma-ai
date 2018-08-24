###################################################################################################
#
# EnergyLoss.py
#
# Copyright (C) by Andreas Zoglauer & Winnie Lee.
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


  def train(self):
    """
    Switch between the various machine-learning libraries based on self.Algorithm
    """ 
    
    if self.Algorithms.startswith("TMVA:"):
      self.trainTMVAMethods()
    elif self.Algorithms.startswith("SKL:"):
      self.trainSKLMethods()
    else:
      print("ERROR: Unknown algorithm: {}".format(self.Algorithms))
    
    return
  
  
###################################################################################################


  def trainSKLMethods(self):
    import time
    print("{}: retrieve from ROOT tree".format(time.time()))

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
        VariableMap[B.GetName()] = array.array('i', [0])
      else:
        VariableMap[B.GetName()] = array.array('f', [0])
      DataTree.SetBranchAddress(B.GetName(), VariableMap[B.GetName()])

    # transform data into numpy array
    import numpy as np
    #total_data=3000
    total_data = min(self.MaxEvents, DataTree.GetEntries())
    
    X_data = np.zeros((total_data, 40)) # space holder
    y_data = np.zeros((total_data, 1))

    all_features = list(VariableMap.keys())
    all_features.remove("SequenceLength")
    all_features.remove("SimulationID")
    all_features.remove("EvaluationIsReconstructable")
    all_features.remove("EvaluationZenithAngle")

    all_features.remove("EvaluationIsCompletelyAbsorbed") #y

    print("{}: start formatting array".format(time.time()))
    
    for x in range(0, total_data):
      
      if x%1000 == 0 and x > 0:
        print("{}: Progress: {}/{}".format(time.time(), x, total_data))
      
      DataTree.GetEntry(x)  # Get row x
      
      new_row=[VariableMap[feature][0] for feature in all_features]
      X_data[x]=np.array(new_row)
      
      if VariableMap["EvaluationIsCompletelyAbsorbed"][0] == 1:
        target=1.0
      else:
        target=0.0
      y_data[x]= target
         
    # remove place holder 
    #y_data = np.delete(y_data, 0)

    print("{}: finish formatting array".format(time.time()))
    
    # alternative: use root_numpy to get data from Root Tree
    """from root_numpy import root2array, rec2array
    branch_names = VariableMap.keys()

    signal = root2array(DataTree, "tree", branch_names)
    signal = rec2array(signal)

    backgr = root2array("", "tree", branch_names)
    backgr = rec2array(backgr)

    # create 2d numpy array for scikit-learn
    X_data = np.concatenate((signal, backgr))
    y_data = np.concatenate((np.ones(signal.shape[0]), np.zeros(backgr.shape[0])))"""
    
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    #from sklearn.cross_validation import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.metrics import classification_report,confusion_matrix
    
    # Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.5, random_state = 0)


    # SVM
    if self.Algorithms == "SKL:SVM":
      print("Running support vector machine ... please stand by...")
      from sklearn.svm import SVC  
    
      svclassifier = SVC(kernel='linear')  
      svclassifier.fit(X_train, y_train)
      y_predicted = svclassifier.predict(X_test) 
      print(svclassifier)
      # print("Training set score: %f" % mlp.score(X_train, y_train))
      # print("Test set score: %f" % mlp.score(X_test, y_test))
      print(confusion_matrix(y_test, y_predicted)) 


    # Run the multi-layer perceptron
    elif self.Algorithms == "SKL:MLP":
      print("Running multi-layer perceptron ... please stand by...")
      
      from sklearn.neural_network import MLPClassifier
      
      mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, activation='logistic', hidden_layer_sizes=(100, 50, 30), random_state=0)
      # MLPClassifier supports only the Cross-Entropy loss function
      # feature scaling
      from sklearn.preprocessing import StandardScaler
      scaler = StandardScaler()
      # Fit only to the training data
      scaler.fit(X_train)
      X_train = scaler.transform(X_train)
      X_test = scaler.transform(X_test)

      mlp.fit(X_train, y_train)  
      y_predicted=mlp.predict(X_test)
      #[coef.shape for coef in mlp.coefs_]
      print(mlp)
      print("Training set score: %f" % mlp.score(X_train, y_train))
      print("Test set score: %f" % mlp.score(X_test, y_test))
      print(confusion_matrix(y_test,y_predicted))
      #print(classification_report(y_test,predictions))


    # Run the random forrest
    elif self.Algorithms == "SKL:RF":
      print("Running random forrest ... please stand by...")

      rf=RandomForestClassifier(n_estimators=1400, criterion ='entropy', random_state=0,bootstrap=False, min_samples_leaf=0.01, max_features='sqrt', min_samples_split=5, max_depth=11)
      rf.fit(X_train, y_train)
      y_predicted = rf.predict(X_test)
    

    # ADABoosting decision tree
    elif self.Algorithms == "SKL:ADABDC":
      print("Running ADABoost'ed decision tree ... please stand by...")

      dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=0.01)
      bdt = AdaBoostClassifier(dt,
                           algorithm='SAMME',
                           n_estimators=800,
                           learning_rate=0.1)
      """from sklearn.model_selection import GridSearchCV
      parameters = {"max_depth":range(3,20),"min_samples_leaf":np.arange(0.01,0.5, 0.03)}
      clf = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=4)
      clf.fit(X=X_data, y=y_data)
      tree_model = clf.best_estimator_
      print (clf.best_score_, clf.best_params_) 
      return"""
      #cross_val_score(clf, iris.data, iris.target, cv=10)
      # train
      print("{}: start training".format(time.time()))
      bdt.fit(X_train, y_train)

      # test
      print("{}: start testing".format(time.time()))
      y_predicted = bdt.predict(X_test)
      
      

      # parameter adjustments
      # - learning rate
      # - scaling? energy value is larger but only around 1k~10k times
      
    else:
      print("ERROR: Unknown algorithm: {}".format(self.Algorithms))
      return
      
    # evaluate (roc curve)
    print(classification_report(y_test, y_predicted, target_names=["background", "signal"]))
    print("Area under ROC curve: %.4f"%(roc_auc_score(y_test, y_predicted)))
    
    
    
###################################################################################################


  def trainTMVAMethods(self):
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
    print("Started training")
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
