###################################################################################################
#
# DecayIdentifcation.py
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


class DecayIdentification:
  """
  This class performs energy loss training. A typical usage would look like this:

  AI = DecayIdentification("Ling2.seq3.quality.root", "Results", "MLP,BDT", 1000000)
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

    total_data = min(self.MaxEvents, DataTree.GetEntries())

    X_data = np.zeros((total_data, 40)) # space holder

    if self.Algorithms.startswith("TF:"):
      y_data = np.zeros((total_data, 2))
    else:
      y_data = np.zeros((total_data, 1))

    all_features = list(VariableMap.keys())
    all_features.remove("SequenceLength")
    all_features.remove("SimulationID")
    all_features.remove("EvaluationIsReconstructable")
    all_features.remove("EvaluationZenithAngle")
    all_features.remove("EvaluationIsDecay")

    all_features.remove("EvaluationIsCompletelyAbsorbed") #y

    print("{}: start formatting array".format(time.time()))

    for x in range(0, total_data):

      if x%1000 == 0 and x > 0:
        print("{}: Progress: {}/{}".format(time.time(), x, total_data))

      DataTree.GetEntry(x)  # Get row x

      new_row=[VariableMap[feature][0] for feature in all_features]
      X_data[x]=np.array(new_row)

      if VariableMap["EvaluationIsDecay"][0] == 1:
        target=1.0
      else:
        target=0.0

      if self.Algorithms.startswith("TF:"):
        y_data[x][0]= target
        y_data[x][1]= 1-target
      else:
        y_data[x]= target

    print("{}: finish formatting array".format(time.time()))



    # Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.5, random_state = 0)
    return X_train, X_test, y_train, y_test


  def trainSKLMethods(self):
    import time
    import numpy as np

    from sklearn import datasets
    #from sklearn.cross_validation import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.metrics import classification_report,confusion_matrix

    # load training and testing data
    X_train, X_test, y_train, y_test = self.loadData()


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

  def trainTFMethods(self):
    import tensorflow as tf
    import numpy as np
    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt
    import time
    X_train, X_test, y_train, y_test = self.loadData()
    # DATA SET PARAMETERS
    # Get our dimensions for our different variables and placeholders:
    numFeatures = X_train.shape[1]
    # numLabels = number of classes we are predicting (here just 2: good or bad)
    numLabels = y_train.shape[1]

    if self.Algorithms == "TF:NN":
      MaxIterations = 500
      # Placeholders
      InputDataSpaceSize=numFeatures
      OutputDataSpaceSize=numLabels
      print("      ... placeholders ...")
      X = tf.placeholder(tf.float32, [None, InputDataSpaceSize], name="X")
      Y = tf.placeholder(tf.float32, [None, OutputDataSpaceSize], name="Y")


      # Layers: 1st hidden layer X1, 2nd hidden layer X2, etc.
      print("      ... hidden layers ...")
      H = tf.contrib.layers.fully_connected(X, 10) #, activation_fn=tf.nn.relu6, weights_initializer=tf.truncated_normal_initializer(0.0, 0.1), biases_initializer=tf.truncated_normal_initializer(0.0, 0.1))
      H = tf.contrib.layers.fully_connected(H, 100) #, activation_fn=tf.nn.relu6, weights_initializer=tf.truncated_normal_initializer(0.0, 0.1), biases_initializer=tf.truncated_normal_initializer(0.0, 0.1))
      H = tf.contrib.layers.fully_connected(H, 1000) #, activation_fn=tf.nn.relu6, weights_initializer=tf.truncated_normal_initializer(0.0, 0.1), biases_initializer=tf.truncated_normal_initializer(0.0, 0.1))


      print("      ... output layer ...")
      Output = tf.contrib.layers.fully_connected(H, OutputDataSpaceSize, activation_fn=None)

      # Loss function
      print("      ... loss function ...")
      #LossFunction = tf.reduce_sum(np.abs(Output - Y)/TestBatchSize)
      #LossFunction = tf.reduce_sum(tf.pow(Output - Y, 2))/TestBatchSize
      LossFunction = tf.nn.l2_loss(Output-Y, name="squared_error_cost")

      # Minimizer
      print("      ... minimizer ...")
      Trainer = tf.train.AdamOptimizer().minimize(LossFunction)

      #Accuracy
      # argmax(Y, 1) is the correct label
      correct_predictions_OP = tf.equal(tf.argmax(Output,1),tf.argmax(Y,1))
      accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))

      # Create and initialize the session
      print("      ... session ...")
      sess = tf.Session()
      sess.run(tf.global_variables_initializer())

      print("      ... writer ...")
      writer = tf.summary.FileWriter("OUT_ToyModel2DGauss", sess.graph)
      writer.close()

      # Add ops to save and restore all the variables.
      print("      ... saver ...")
      Saver = tf.train.Saver()



      ###################################################################################################
      # Step 3: Training and evaluating the network
      ###################################################################################################


      print("Info: Training and evaluating the network")

      # Train the network
      #Timing = time.process_time()

      TimesNoImprovement = 0
      BestMeanSquaredError = sys.float_info.max

      def CheckPerformance():
        global TimesNoImprovement
        global BestMeanSquaredError

        MeanSquaredError = sess.run(tf.nn.l2_loss(Output - y_test),  feed_dict={X: X_test})

        print("Iteration {} - MSE of test data: {}".format(Iteration, MeanSquaredError))
        print("final accuracy on test set: %s" %str(sess.run(accuracy_OP, feed_dict={X: X_test, Y: y_test})))


      # Main training and evaluation loop

      for Iteration in range(0, MaxIterations):
        # Take care of Ctrl-C
        #if Interrupted == True: break

        # Train
        sess.run(Trainer, feed_dict={X: X_train, Y: y_train})

        # Check performance: Mean squared error
        if Iteration > 0 and Iteration % 20 == 0:
          CheckPerformance()

        if TimesNoImprovement == 100:
          print("No improvement for 30 rounds")
          break;

    # logistic regression
    elif self.Algorithms == "TF:LR":



      # TRAINING SESSION PARAMETERS
      # number of times we iterate through training data
      # tensorboard shows that accuracy plateaus at ~25k epochs
      numEpochs = 2700
      # a smarter learning rate for gradientOptimizer
      learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                                global_step= 1,
                                                decay_steps=X_train.shape[0],
                                                decay_rate= 0.95,
                                                staircase=True)

      # tensors: placeholders
      # X = X-matrix / feature-matrix / data-matrix... It's a tensor to hold our
      # data. 'None' here means that we can hold any number of emails
      X = tf.placeholder(tf.float32, [None, numFeatures])
      # yGold = Y-matrix / label-matrix / labels... This will be our correct answers matrix.
      yGold = tf.placeholder(tf.float32, [None, numLabels])


      # tensors: weights and bias term for regression
      # Values are randomly sampled from a Gaussian with a standard deviation of:
      #     sqrt(6 / (numInputNodes + numOutputNodes + 1))

      weights = tf.Variable(tf.random_normal([numFeatures,numLabels],
                                             mean=0,
                                             stddev=(np.sqrt(6/numFeatures+
                                                               numLabels+1)),
                                             name="weights"))

      bias = tf.Variable(tf.random_normal([1,numLabels],
                                          mean=0,
                                          stddev=(np.sqrt(6/numFeatures+numLabels+1)),
                                          name="bias"))

      ######################
      ### PREDICTION OPS ###
      ######################

      # INITIALIZE our weights and biases
      init_OP = tf.global_variables_initializer()

      # PREDICTION ALGORITHM i.e. FEEDFORWARD ALGORITHM
      apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
      add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
      activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")

      #####################
      ### EVALUATION OP ###
      #####################

      # COST FUNCTION i.e. MEAN SQUARED ERROR
      cost_OP = tf.nn.l2_loss(activation_OP-yGold, name="squared_error_cost")


      #######################
      ### OPTIMIZATION OP ###
      #######################

      # OPTIMIZATION ALGORITHM i.e. GRADIENT DESCENT
      training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)


      # visualization
      epoch_values=[]
      accuracy_values=[]
      cost_values=[]
      # Turn on interactive plotting
      plt.ion()
      # Create the main, super plot
      fig = plt.figure()
      # Create two subplots on their own axes and give titles
      ax1 = plt.subplot("211")
      ax1.set_title("TRAINING ACCURACY", fontsize=18)
      ax2 = plt.subplot("212")
      ax2.set_title("TRAINING COST", fontsize=18)
      plt.tight_layout()
      #####################
      ### RUN THE GRAPH ###
      #####################

      # Create a tensorflow session
      sess = tf.Session()

      # Initialize all tensorflow variables
      sess.run(init_OP)

      ## Ops for vizualization
      # argmax(activation_OP, 1) gives the label our model thought was most likely
      # argmax(yGold, 1) is the correct label
      correct_predictions_OP = tf.equal(tf.argmax(activation_OP,1),tf.argmax(yGold,1))
      # False is 0 and True is 1, what was our average?
      accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))
      # Summary op for regression output
      activation_summary_OP = tf.summary.histogram("output", activation_OP)
      # Summary op for accuracy
      accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)
      # Summary op for cost
      cost_summary_OP = tf.summary.scalar("cost", cost_OP)
      # Summary ops to check how variables (W, b) are updating after each iteration
      weightSummary = tf.summary.histogram("weights", weights.eval(session=sess))
      biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))
      # Merge all summaries
      all_summary_OPS = tf.summary.merge_all()
      # Summary writer
      writer = tf.summary.FileWriter("summary_logs", sess.graph)

      # Initialize reporting variables
      cost = 0
      diff = 1

      # Training epochs
      for i in range(numEpochs):
        if i > 1 and diff < .0001:
          print("change in cost %g; convergence."%diff)
          break
        else:
          # Run training step
          step = sess.run(training_OP, feed_dict={X: X_train, yGold: y_train})
          # Report occasional stats
          if i % 10 == 0:
            # Add epoch to epoch_values
            epoch_values.append(i)
            # Generate accuracy stats on test data
            summary_results, train_accuracy, newCost = sess.run(
                [all_summary_OPS, accuracy_OP, cost_OP],
                feed_dict={X: X_train, yGold: y_train}
            )
            # Add accuracy to live graphing variable
            accuracy_values.append(train_accuracy)
            # Add cost to live graphing variable
            cost_values.append(newCost)
            # Write summary stats to writer
            writer.add_summary(summary_results, i)
            # Re-assign values for variables
            diff = abs(newCost - cost)
            cost = newCost

            #generate print statements
            print("step %d, training accuracy %g"%(i, train_accuracy))
            print("step %d, cost %g"%(i, newCost))
            print("step %d, change in cost %g"%(i, diff))

            # Plot progress to our two subplots
            accuracyLine, = ax1.plot(epoch_values, accuracy_values)
            costLine, = ax2.plot(epoch_values, cost_values)
            fig.canvas.draw()
            time.sleep(1)


      # How well do we perform on held-out test data?
      print("final accuracy on test set: %s" %str(sess.run(accuracy_OP,
                                                           feed_dict={X: X_test,
                                                                      yGold: y_test})))

      ##############################
      ### SAVE TRAINED VARIABLES ###
      ##############################

      # Create Saver
      saver = tf.train.Saver()
      # Save variables to .ckpt file
      # saver.save(sess, "trained_variables.ckpt")

      # Close tensorflow session
      sess.close()
    return


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

    SignalCut = ROOT.TCut("EvaluationIsDecay >= 0.5")
    BackgroundCut = ROOT.TCut("EvaluationIsDecay < 0.5")
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
