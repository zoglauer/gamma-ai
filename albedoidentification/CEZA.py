
###################################################################################################
#
# Classifciation Evaluation Zenith Angle
#
# Copyright (C) by Andreas Zoglauer, Jasper Gan, & Joan Zhu.
# All rights reserved.
#
# Please see the file License.txt in the main repository for the copyright-notice.
#
###################################################################################################




###################################################################################################

""" TMVA imports """
import ROOT
import array
import sys

""" Tensorflow imports """
import tensorflow as tf
import numpy as np
import random
import time


###################################################################################################


class CEZA:

  """
  This class performs classification on evaulation of whether isReconstructable and isAbsorbed.
  A typical usage would look like this:

  AI = EnergyLossIdentification("Ling2.seq3.quality.root", "Results", "MLP,BDT", 1000000)
  AI.train()
  AI.test()"""


  def __init__(self, FileName, Output, Algorithm, MaxEvents, Quality):
    self.FileName = FileName
    self.OutputPrefix = Output
    self.Algorithms = Algorithm
    self.MaxEvents = MaxEvents
    self.Quality = Quality


###################################################################################################


  def train(self):
    """
    Switch between the various machine-learning libraries based on self.Algorithm
    """

    if self.Algorithms.startswith("TMVA:"):
     self.trainTMVAMethods()
    elif self.Algorithms.startswith("TF:"):
      self.trainTFMethods()
    else:
     print("ERROR: Unknown algorithm: {}".format(self.Algorithms))

    return


###################################################################################################


  def loadData(self):
    """
    Prepare numpy array dataset for scikit-learn and tensorflow models
    """

    import time
    import numpy as np
    from sklearn.model_selection import train_test_split

    print("{}: retrieve from ROOT tree".format(time.time()))

    # Open the file
    DataFile = ROOT.TFile(self.FileName)
    if DataFile.IsOpen() == False:
      print("Error opening data file")
      return False

    # Get the data tree
    DataTree = DataFile.Get("Quality")
    if DataTree is None:
      print("Error reading data tree from root file")
      return False

    Branches = DataTree.GetListOfBranches()

    VariableMap = {}

    # Create a map of the branches, i.e. the columns
    for B in list(Branches):
      if B.GetName() == "EvaluationZenithAngle":
        VariableMap[B.GetName()] = array.array('f', [0])
      else:
        VariableMap[B.GetName()] = array.array('f', [0])
      DataTree.SetBranchAddress(B.GetName(), VariableMap[B.GetName()])

    # transform data into numpy array

    total_data = min(self.MaxEvents, DataTree.GetEntries())

    #placeholders
    X_data = np.zeros((total_data, 40))
    # if self.Algorithms.startswith("TF:"):
    #   y_data = np.zeros((total_data, 2))
    # else:
    y_data = np.zeros((total_data, 1))

    all_features = list(VariableMap.keys())
    all_features.remove("SequenceLength")
    all_features.remove("SimulationID")
    all_features.remove("EvaluationIsReconstructable")
    all_features.remove("EvaluationIsCompletelyAbsorbed")

    all_features.remove("EvaluationZenithAngle")

    print("{}: start formatting array".format(time.time()))

    for x in range(total_data):

      if x%1000 == 0 and x > 0:
        print("{}: Progress: {}/{}".format(time.time(), x, total_data))

      DataTree.GetEntry(x)  # Get row x

      new_row=[VariableMap[feature][0] for feature in all_features]
      X_data[x]=np.array(new_row)

      if VariableMap["EvaluationZenithAngle"][0] < 90:
        target = 1.0
      else:
        target = 0.0

      #print("{0} vs. {1}".format(target, VariableMap["EvaluationZenithAngle"][0]))
      
      # if self.Algorithms.startswith("TF:"):
      #   y_data[x][0]= target
      #   y_data[x][1]= 1-target
      # else:
      y_data[x]= target

    print("{}: finish formatting array".format(time.time()))

    # Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state = 42)

    print("Data size: training: {0}, testing: {1}".format(len(y_train), len(y_test)))

    return X_train, X_test, y_train, y_test




###################################################################################################

  def trainTFMethods(self):
    """
    Main training function that runs methods through Tensorflow library

    Returns
    -------
    bool
      True is everything went well, False in case of error
    """

    ###################################################################################################
    # Step 1: Load data and perform train-test split
    ###################################################################################################

    #TODO:  Stored retrieved + prepared data in file to avoid rerunning this step every time
    XTrain, XTest, YTrain, YTest = self.loadData()

    ###################################################################################################
    # Step 2: Input parameters
    ###################################################################################################

    # Input parameters
    print("\nInfo: Preparing input parameters...")
    #TODO: Change these paramters & create validation set out of training set

    # SubBatchSize = len(XTrain) // 2        # num events in testing data = 1110
    # print("SUB BATCH SIZE: ", SubBatchSize)
    #
    # NTrainingBatches = 1
    # TrainingBatchSize = NTrainingBatches*SubBatchSize
    #
    # NTestingBatches = 1
    # TestBatchSize = NTestingBatches*SubBatchSize

    TotalData = XTrain.shape[0] # = 5182

    SplitSize = int(XTrain.shape[0]*0.7) #Split size of 70:30 for training and validation set

    print(YTrain)
    print(np.sum(YTrain))

    XTrain, XVal = XTrain[:SplitSize], XTrain[SplitSize:]
    YTrain, YVal = YTrain[:SplitSize], YTrain[SplitSize:]

    print("Total Train Data: ", TotalData)
    print("X_Val: ", XVal.shape) # (1555,40)
    print("Y_Val: ", YVal.shape) # (1555,2)
    print("X_Train: ", XTrain.shape) # (3627,40)
    print("Y_Train: ", YTrain.shape) # (3627,2)

    SubBatchSize = int(XTrain.shape[0]*.3)

    Interrupted = False

    MaxIterations = 10000
    LearningRate = 0.01



    ###################################################################################################
    # Setting up the neural network
    ###################################################################################################


    print("\nInfo: Setting up Tensorflow neural network...")

    # Placeholders
    print("      ... placeholders ...")
    # shape as None = variable length of data points, NumFeatures
    X = tf.placeholder(tf.float32, [None, XTrain.shape[1]], name="X")
    Y = tf.placeholder(tf.float32, [None, YTrain.shape[1]], name="Y")
    keep_prob = tf.placeholder("float")

    # Layers: 1st hidden layer X1, 2nd hidden layer X2, etc.
    print("      ... hidden layers ...")
    H = tf.contrib.layers.fully_connected(X, 20) #, activation_fn=tf.nn.relu6, weights_initializer=tf.truncated_normal_initializer(0.0, 0.1), biases_initializer=tf.truncated_normal_initializer(0.0, 0.1))
    # H = tf.contrib.layers.fully_connected(H, 100)
    # H = tf.contrib.layers.fully_connected(H, 1000)

    #TODO: Add Dropout to reduce overfitting

    # TODO: Rename output to network...
    print("      ... output layer ...")
    Output = tf.contrib.layers.fully_connected(H, YTrain.shape[1], activation_fn=None)


    # Loss function
    print("      ... loss function ...")
    # AZ: LossFunction = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Output))
    LossFunction = tf.reduce_sum(tf.pow(Output - Y, 2))

    # Minimizer
    print("      ... minimizer ...")
    Trainer = tf.train.AdamOptimizer(learning_rate = LearningRate).minimize(LossFunction)

    # Create and initialize the session
    print("      ... session ...")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    print("      ... writer ...")
    writer = tf.summary.FileWriter("OUT_ToyModel2DGauss", sess.graph) # ???
    writer.close()

    # Add ops to save and restore all the variables.
    print("      ... saver ...")
    Saver = tf.train.Saver()

    ###################################################################################################
    # Training and evaluating the network
    ###################################################################################################

    print("\nInfo: Training and evaluating the network")

    # Train the network
    Timing = time.process_time()

    # TimesNoImprovement = 0
    BestError = sys.float_info.max

    def CheckPerformance():
      # nonlocal TimesNoImprovement
      # nonlocal BestError

      Error = sess.run(LossFunction, feed_dict={X: XVal, Y: YVal})/YVal.size

      # print("Iteration {} - Error of validation data: {}".format(Iteration, Error))
      print("\tAverage deviation of validation data: {}".format(Error))
      
      pred_temp = tf.equal(tf.round(Output), tf.round(Y))
      accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
      
      print("\tValidation accuracy: {0}%".format(100*sess.run(accuracy,feed_dict={X:XVal,Y:YVal})))
      # if BestError - Error > 0.0001:
      #   BestError = Error
      #   TimesNoImprovement = 0
      #
      # else: # don't iterate if difference is too small
      #   TimesNoImprovement += 1


    # Main training and evaluation loop

    for Iteration in range(MaxIterations):
      # Take care of Ctrl-C
      if Interrupted == True: break

      for Batch in range(int(XTrain.shape[0]/SubBatchSize)):
        if Interrupted == True: break

        Start = Batch * SubBatchSize    # SubBatchSize = TotalData * TestSize
        Stop = (Batch + 1) * SubBatchSize
        # print("Start: ", Start)
        # print("Stop: ", Stop)
        BatchX = XTrain[Start:Stop]
        BatchY = YTrain[Start:Stop]

        _, Loss = sess.run([Trainer, LossFunction], feed_dict={X: BatchX, Y: BatchY})

######################
      # _, Loss = sess.run([Trainer, LossFunction], feed_dict=({X:XTrain, Y:YTrain}))
######################

      if Iteration > 0 and Iteration % 200 == 0:
        CheckPerformance()
        print("Iteration {} - Error of train data: {}".format(Iteration, Loss))

      # if TimesNoImprovement == 10000:
      #   print("No improvement for 10000 rounds")
      #   break;

    print("\n\tTraining Complete!\n")

    Timing = time.process_time() - Timing
    if Iteration > 0:
      print("Time per training loop: ", Timing/Iteration, " seconds")

    # print("Error: " + str(BestError))

    correct_predictions_OP = tf.equal(tf.argmax(Output,1),tf.argmax(Y,1))
    # correct_predictions_OP = tf.equal(tf.cast(Output > 0, tf.float32), Y)
    accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))

    print("Final Test Accuracy: {}".format(sess.run(accuracy_OP, feed_dict={X: XTest, Y: YTest})))

    input("Press [enter] to EXIT")
    sys.exit(0)




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
    ResultsFile = ROOT.TFile(FullPrefix + ".root","RECREATE")

    Factory = ROOT.TMVA.Factory("TMVAClassification", ResultsFile,
                                ":".join([
                                    "!V",
                                    "!Silent",
                                    "Color",
                                    "DrawProgressBar",
                                    "Transformations=I;D;P;G,D",
                                    "AnalysisType=Classification"]
                                         ))

    DataLoader = ROOT.TMVA.DataLoader("Results")

    IgnoredBranches = [ 'SimulationID', 'SequenceLength']  #'EvaluationZenithAngle', 'EvaluationIsReconstructable', 'EvaluationIsCompletelyAbsorbed']
    Branches = DataTree.GetListOfBranches()

    for Name in IgnoredBranches:
       DataLoader.AddSpectator(Name, "F")

    for b in list(Branches):
        if not b.GetName() in IgnoredBranches:
            if not b.GetName().startswith("Evaluation"):
                DataLoader.AddVariable(b.GetName(), "F")

    SignalCut = ROOT.TCut("EvaluationZenithAngle > 90")
    BackgroundCut = ROOT.TCut("EvaluationZenithAngle <= 90")
    DataLoader.SetInputTrees(DataTree, SignalCut, BackgroundCut)

    DataLoader.PrepareTrainingAndTestTree(SignalCut,
                                       BackgroundCut,
                                       ":".join([
                                            "nTrain_Signal=0",
                                            "nTrain_Background=0",
                                            "SplitMode=Random",
                                            "NormMode=NumEvents",
                                            "!V"
                                           ]))


    # Neural Networks
    if 'MLP' in self.Algorithms:
      method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kMLP, "MLP",
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
    if 'PDEFoamBoost' in self.Algorithms:
      method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kPDEFoam, "PDEFoamBoost",
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

    # PDERSPCA
    if 'PDERSPCA' in self.Algorithms:
      method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kPDERS, "PDERSPCA",
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
    if 'BDT' in self.Algorithms:
      method = Factory.BookMethod(DataLoader, ROOT.TMVA.Types.kBDT, "BDT",
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

    # Finally test, train & Algorithm all methods
    Factory.TrainAllMethods()
    Factory.TestAllMethods()
    Factory.EvaluateAllMethods()



    reader = ROOT.TMVA.Reader("!Color:!Silent");
    variablemap = {}

    for Name in IgnoredBranches:
      variablemap[Name] = array.array('f', [0])
      DataTree.SetBranchAddress(Name, variablemap[Name])
      reader.AddSpectator(Name, variablemap[Name])

    for B in list(Branches):
      if not B.GetName() in IgnoredBranches:
        if not B.GetName().startswith("Evaluation"):
          variablemap[B.GetName()] = array.array('f', [0])
          reader.AddVariable(B.GetName(), variablemap[B.GetName()])
          DataTree.SetBranchAddress(B.GetName(), variablemap[B.GetName()])
          print("Added: " + B.GetName())

    for B in list(Branches):
      if B.GetName().startswith("EvaluationZenithAngle"):
        variablemap[B.GetName()] = array.array('f', [0])
        DataTree.SetBranchAddress(B.GetName(), variablemap[B.GetName()])


    # TODO: loop over different readers that call different methods and output best one
    Algorithm = ''
    if 'MLP' in self.Algorithms:
      Algorithm = 'MLP'
      reader.BookMVA("MLP","Results/weights/TMVAClassification_MLP.weights.xml")
    elif 'BDT' in self.Algorithms:
      Algorithm = 'BDT'
      reader.BookMVA("BDT","Results/weights/TMVAClassification_BDT.weights.xml")
    elif 'PDEFoamBoost' in self.Algorithms:
      Algorithm = 'PDEFoamBoost'
      reader.BookMVA("PDEFoamBoost","Results/weights/TMVAClassification_PDEFoamBoost.weights.xml")
    elif 'PDERSPCA' in self.Algorithms:
      Algorithm = 'PDERSPCA'
      reader.BookMVA("PDERSPCA","Results/weights/TMVAClassification_PDERSPCA.weights.xml")

    NEvents = 0
    NGoodEvents = 0

    NLearnedGoodEvents = 0
    NLearnedCorrectEvents = 0

    varx = array.array('f',[0]) #; reader.AddVariable("EvaluationZenithAngle",varx)
    vary = array.array('f',[0]) #; reader.AddVariable("result",vary)

    for x in range(0, min(100, DataTree.GetEntries())):
      DataTree.GetEntry(x)

      NEvents += 1

      print("\nSimulation ID: " + str(int(variablemap["SimulationID"][0])) + ":")

      result = reader.EvaluateMVA(Algorithm)
      vary.append(result)

      r = 2
      IsGood = True
      IsGoodThreshold = 0.2

      IsLearnedGood = True
      IsLearnedGoodThreshold = 0.06 # Adjust this as see fit

      for b in list(Branches):
        Name = b.GetName()

        if Name.startswith("EvaluationZenithAngle"):
          print(Name + " " + str(variablemap[Name][0]) + " vs. " + str(90))
          varx.append(variablemap[Name][0])
          if abs(variablemap[Name][0] - 90 > IsGoodThreshold):
            IsGood = False
          r += 1

      if IsGood == True:
        NGoodEvents += 1
        print(" --> Good event")
      else:
        print(" --> Bad event")

      if (IsLearnedGood == True and IsGood == True) or (IsLearnedGood == False and IsGood == False):
        NLearnedCorrectEvents += 1

    print("\nResult:")
    print("All events: " + str(NEvents))
    print("Good events: " + str(NGoodEvents))
    print("Correctly identified: " + str(NLearnedCorrectEvents / NEvents))

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
            bdtOutput = reader.EvaluateMVA(Algorithm)

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

    ROOT.TestTree.Draw(Algorithm + ">>hSig(22,-1.1,1.1)","classID == 0","goff")  # signal
    ROOT.TestTree.Draw(Algorithm + ">>hBg(22,-1.1,1.1)","classID == 1", "goff")  # background

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
