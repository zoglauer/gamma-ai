
###################################################################################################
#
# Classification Evaluation isReconstructable isAbsorbed
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
import matplotlib.pyplot as plt
import numpy as np
import random
import time


###################################################################################################


class CERA:

  """
  This class performs classification on evaulation of whether isReconstructable and isAbsortbed.
  A typical usage would look like this:

  AI = EnergyLossIdentification("Ling2.seq3.quality.root", "Results", "MLP,BDT", 1000000)
  AI.train()
  AI.test()"""

  def __init__(self, Filename, Output, Algorithm, MaxEvents, Quality):
    self.Filename = Filename
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
    # elif self.Algorithms.startswith("SKL:"):
    #   self.trainSKLMethods()
    elif self.Algorithms.startswith("TF:"):
      self.trainTFMethods()
    else:
      print("ERROR: Unknown algorithm: {}".format(self.Algorithms))

    return



  def trainTFMethods(self):
    """
    Main training function that runs methods through Tensorflow library

    Returns
    -------
    bool
      True is everything went well, False in case of error
    """

    ###################################################################################################
    # Step 1: Reading data
    ###################################################################################################

    # Open the file
    DataFile = ROOT.TFile(self.Filename)
    if DataFile.IsOpen() == False:
      print("Error opening data file")
      return False

    # Get the data tree
    DataTree = DataFile.Get("Quality")
    if DataTree is None:
      print("Error reading data tree from root file")
      return False

    # Reading training dataset
    DataLoader = ROOT.TMVA.DataLoader("Results")
    Branches = list(DataTree.GetListOfBranches())

    # Create a map of the branches
    VariableMap = {}

    for B in Branches:
      if B.GetName() == "EvaluationIsReconstructable":
        VariableMap[B.GetName()] = array.array('i', [0])
      else:
        VariableMap[B.GetName()] = array.array('f', [0])
      DataTree.SetBranchAddress(B.GetName(), VariableMap[B.GetName()])


    AllFeatures = list(VariableMap.keys())
    AllFeatures.remove("SequenceLength")
    AllFeatures.remove("SimulationID")
    AllFeatures.remove("EvaluationZenithAngle")
    AllFeatures.remove("EvaluationIsCompletelyAbsorbed")

    YTarget = "EvaluationIsReconstructable"
    AllFeatures.remove(YTarget)

    XEventDataBranches = [B for B in Branches
          if not (B.GetName().startswith("Evaluation") or B.GetName().startswith("SimulationID")
                  or B.GetName().startswith("SequenceLength"))]

    YResultBranches = [B for B in Branches
                      if B.GetName().startswith("EvaluationIsReconstructable")]

    print("Eval Reconstructible: ", YResultBranches[0])

    ###################################################################################################
    # Step 2: Input parameters
    ###################################################################################################

    # Input parameters
    TotalData = min(self.MaxEvents, DataTree.GetEntries())

    # Ensure TotalData evenly splittable so we can split half into training and half into testing
    if TotalData % 2 == 1:
      TotalData -= 1

    SubBatchSize = TotalData // 2        # num events in testing data

    NTrainingBatches = 1
    TrainingBatchSize = NTrainingBatches*SubBatchSize

    NTestingBatches = 1
    TestBatchSize = NTestingBatches*SubBatchSize

    Interrupted = False

    ###################################################################################################
    # Step 3: Construct training and testing dataset
    ###################################################################################################

    # Transform data into numpy array

    XTrain = np.zeros((TotalData // 2, len(XEventDataBranches)))
    XTest = np.zeros((TotalData // 2, len(XEventDataBranches)))
    YTrain = np.zeros((TotalData // 2, len(YResultBranches)))
    YTest = np.zeros((TotalData // 2, len(YResultBranches)))

    for i in range(TotalData):
      # Print final progress
      if i == TotalData - 1:
        print("{}: Progress: {}/{}".format(time.time(), i + 1, TotalData))

      # Display progress throughout
      elif i % 1000 == 0:
        print("{}: Progress: {}/{}".format(time.time(), i, TotalData))

      DataTree.GetEntry(i) # ???

      NewRow = [VariableMap[feature][0] for feature in AllFeatures]

      # Split half the X data into training set and half into testing set
      if i % 2 == 0:
        XTrain[i // 2] = np.array(NewRow)
        YTrain[i // 2] = float(VariableMap[YTarget][0])
      else:
        XTest[i // 2] = np.array(NewRow)
        YTest[i // 2] = float(VariableMap[YTarget][0])

    print("{}: finish formatting array".format(time.time()))

    ###################################################################################################
    # Setting up the neural network
    ###################################################################################################

    print("Info: Setting up Tensorflow neural network...")

    # Placeholders
    print("      ... placeholders ...")
    # shape as None = variable length of data points, NumFeatures
    X = tf.placeholder(tf.float32, [None, XTrain.shape[1]], name="X")
    Y = tf.placeholder(tf.float32, [None, YTrain.shape[1]], name="Y")

    # Layers: 1st hidden layer X1, 2nd hidden layer X2, etc.
    print("      ... hidden layers ...")
    H = tf.contrib.layers.fully_connected(X, 20) #, activation_fn=tf.nn.relu6, weights_initializer=tf.truncated_normal_initializer(0.0, 0.1), biases_initializer=tf.truncated_normal_initializer(0.0, 0.1))
    # H = tf.contrib.layers.fully_connected(H, 100)
    # H = tf.contrib.layers.fully_connected(H, 1000)

    print("      ... output layer ...")
    Output = tf.contrib.layers.fully_connected(H, len(YResultBranches), activation_fn=None)

    print("      ... loss function ...")
    # Loss function sigmoid cross entropy with logits to be used here because
    # sigmoid cross entropy runs a binary classification (true or false).
    # Since our data is either signal or not signal (true or false), this results
    # in predicting a mutually exclusive label for just one class. Logits are the
    # normalized output of the neural net (between 0 and 1).
    LossFunction = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Output))

    # ROC visualization
    plot_points = tf.contrib.metrics.streaming_curve_points(labels=Y, predictions=tf.nn.sigmoid(Output))

    # Minimizer
    print("      ... minimizer ...")
    Trainer = tf.train.AdamOptimizer().minimize(LossFunction)

    # Create and initialize the session -- all variables and operations should be
    # initalized above this line
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

    print("Info: Training and evaluating the network")

    # Train the network
    Timing = time.process_time()

    TimesNoImprovement = 0
    BestError = sys.float_info.max

    def CheckPerformance():
      nonlocal TimesNoImprovement
      nonlocal BestError

      Error = sess.run(LossFunction, feed_dict={X: XTest, Y: YTest})

      print("Iteration {} - Error of test data: {}".format(Iteration, Error))

      if BestError - Error > 0.0001:
        BestError = Error
        TimesNoImprovement = 0

      else: # don't iterate if difference is too small
        TimesNoImprovement += 1

    # Main training and evaluation loop
    MaxIterations = 50000
    for Iteration in range(0, MaxIterations):
      # Take care of Ctrl-C
      if Interrupted == True: break

      # Train
      for Batch in range(0, NTrainingBatches):
        if Interrupted == True: break

        Start = Batch * SubBatchSize
        Stop = (Batch + 1) * SubBatchSize
        _, Loss = sess.run([Trainer, LossFunction], feed_dict={X: XTrain[Start:Stop], Y: YTrain[Start:Stop]})

      # Check performance: Mean squared error
      if Iteration > 0 and Iteration % 200 == 0:
        CheckPerformance()
        print("Iteration {} - Error of train data: {}".format(Iteration, Loss))

      if TimesNoImprovement == 10:
        print("No improvement for 10 rounds")
        break;

    Timing = time.process_time() - Timing
    if Iteration > 0:
      print("Time per training loop: ", Timing/Iteration, " seconds")

    # Reporting accuracy and error
    print("Error: " + str(BestError))

    """ With a loss function of tf.sigmoid_cross_entropy_with_logits, this will output a logistic
    curve such that predictions of Output > 0 are predictions with a greater than 50% chance of
    being correct based off the training data. We cast these boolean results of Output > 0 to floats
    and then check how many are equal to Y to find the number of correct predictions.
    """
    correct_predictions_OP = tf.equal(tf.cast(Output > 0, tf.float32), Y)
    accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))
    print("Final accuracy on test set: %s" %str(sess.run(accuracy_OP, feed_dict={X: XTest, Y: YTest})))

    # ROC visualization
    print("ROC visualization")
    # print((sess.run(plot_points, feed_dict={X: XTest, Y: YTest})[0]))
    plt.imshow(sess.run(plot_points, feed_dict={X: XTest, Y: YTest})[0])
    plt.show()

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
    DataFile = ROOT.TFile(self.Filename)
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

    DataLoader = ROOT.TMVA.DataLoader(self.OutputPrefix)

    IgnoredBranches = [ 'SimulationID', 'SequenceLength']
    Branches = DataTree.GetListOfBranches()

    for Name in IgnoredBranches:
        DataLoader.AddSpectator(Name, "F")

    for B in list(Branches):
        if not B.GetName() in IgnoredBranches:
            if not B.GetName().startswith("Evaluation"):
                DataLoader.AddVariable(B.GetName(), "F")

    SignalCut = ROOT.TCut("EvaluationIsReconstructable >= 0.5")
    BackgroundCut = ROOT.TCut("EvaluationIsReconstructable < 0.5")
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


    print("Started training")
    Factory.TrainAllMethods()
    Factory.TestAllMethods()
    Factory.EvaluateAllMethods()

    reader = ROOT.TMVA.Reader("!Color:!Silent");
    variablemap = {}

    for name in IgnoredBranches:
      variablemap[name] = array.array('f', [0])
      DataTree.SetBranchAddress(name, variablemap[name])
      reader.AddSpectator(name, variablemap[name])

    for b in list(Branches):
      if not b.GetName() in IgnoredBranches:
        if not b.GetName().startswith("Evaluation"):
          variablemap[b.GetName()] = array.array('f', [0])
          reader.AddVariable(b.GetName(), variablemap[b.GetName()])
          DataTree.SetBranchAddress(b.GetName(), variablemap[b.GetName()])
          print("Added: " + b.GetName())

    for b in list(Branches):
      if b.GetName().startswith("EvaluationIsReconstructable") or b.GetName().startswith("EvaluationIsCompletelyAbsorbed"):
        variablemap[b.GetName()] = array.array('f', [0])
        DataTree.SetBranchAddress(b.GetName(), variablemap[b.GetName()])

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

    for x in range(0, min(500, DataTree.GetEntries())):
      DataTree.GetEntry(x)

      NEvents += 1

      print("\nSimulation ID: " + str(int(variablemap["SimulationID"][0])) + ":")

      result = reader.EvaluateMVA(Algorithm)
      print(result)
      vary.append(result)

      r = 2
      IsGood = True
      IsGoodThreshold = 0.2

      IsLearnedGood = True
      IsLearnedGoodThreshold = 0.06 # Adjust this as see fit

      for b in list(Branches):
        name = b.GetName()

        if name.startswith("EvaluationIsReconstructable") or name.startswith("EvaluationIsCompletelyAbsorbed"):
          print(name + " " + str(variablemap[name][0]))
          if not variablemap[name][0]:
            IsGood = False
          if result < IsLearnedGoodThreshold:
            IsLearnedGood = False
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

    gcSaver = []

    gcSaver.append(ROOT.TCanvas())

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
