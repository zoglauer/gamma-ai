###################################################################################################
#
# EventTypeIdentification.py
#
# Copyright (C) by Andreas Zoglauer, Anna Shang, Amal Metha & Caitlyn Chen.
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
import random
import time
import collections
import numpy as np
import math, datetime
from voxnet import *
#from volumetric_data import ShapeNet40Vox30


###################################################################################################


class EventTypeIdentification:
  """
  This class performs energy loss training. A typical usage would look like this:

  AI = EventTypeIdentification("Ling2.seq3.quality.root", "Results", "TF:VOXNET", 1000000)
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
    self.Output = 'Results'
    if Output != '':
      self.Output = self.Output + '_' + Output
    self.Algorithms = Algorithm
    self.MaxEvents = MaxEvents

    self.EventTypes = []
    self.EventHits = []
    self.EventTypesTrain = []
    self.EventTypesTest = []
    self.EventHitsTrain = []
    self.EventHitsTest = []
    self.LastEventIndex = 0
    
    self.BatchSize = 20
    self.XBins = 110
    self.YBins = 110
    self.ZBins = 48
    self.MaxLabel = 0

    #might have to tune these values
    self.XMin = -55
    self.XMax = 55

    self.YMin = -55
    self.YMax = 55

    self.ZMin = 0
    self.ZMax = 48
    
    #keras model development
    self.OutputDirectory = "output.txt"
    self.train_test_split = 0.9
    self.keras_model = None


###################################################################################################


  def train(self):
    """
    Switch between the various machine-learning libraries based on self.Algorithm
    """

    #if self.Algorithms.startswith("TF:"):
    #  self.trainTFMethods()
    #elif self.Algorithms.startswith("KERAS:"):
    self.trainKerasMethods()
    #  self.trainTMVAMethods()
    #elif self.Algorithms.startswith("SKL:"):
    #  self.trainSKLMethods()
    #else:
    #  print("ERROR: Unknown algorithm: {}".format(self.Algorithms))

    return


###################################################################################################


  def loadData(self):
    """
    Prepare numpy array datasets for scikit-learn and tensorflow models
    
    Returns:
      list: list of the events types in numerical form: 1x: Compton event, 2x pair event, with x the detector (0: passive material, 1: tracker, 2: absober)
      list: list of all hits as a numpy array containing (x, y, z, energy) as row 
    """
   
    print("{}: Load data from sim file".format(time.time()))


    import ROOT as M

    # Load MEGAlib into ROOT
    M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")

    # Initialize MEGAlib
    G = M.MGlobal()
    G.Initialize()
    
    # Fixed for the time being
    GeometryName = "$(MEGALIB)/resource/examples/geomega/GRIPS/GRIPS.geo.setup"

    # Load geometry:
    Geometry = M.MDGeometryQuest()
    if Geometry.ScanSetupFile(M.MString(GeometryName)) == True:
      print("Geometry " + GeometryName + " loaded!")
    else:
      print("Unable to load geometry " + GeometryName + " - Aborting!")
      quit()
    

    Reader = M.MFileEventsSim(Geometry)
    if Reader.Open(M.MString(self.FileName)) == False:
      print("Unable to open file " + FileName + ". Aborting!")
      quit()

    #Hist = M.TH2D("Energy", "Energy", 100, 0, 600, 100, 0, 600)
    #Hist.SetXTitle("Input energy [keV]")
    #Hist.SetYTitle("Measured energy [keV]")


    EventTypes = []
    EventHits = []

    NEvents = 0
    while True: 
      Event = Reader.GetNextEvent()
      if not Event:
        break
  
      Type = 0
      if Event.GetNIAs() > 0:
        if Event.GetIAAt(1).GetProcess() == M.MString("COMP"):
          Type += 0 + Event.GetIAAt(1).GetDetectorType()
        elif Event.GetIAAt(1).GetProcess() == M.MString("PAIR"):
          Type += 10 + Event.GetIAAt(1).GetDetectorType()
      else:
        break
      
      if Type+1 > self.MaxLabel:
        self.MaxLabel = Type +1
  
      Hits = np.zeros((Event.GetNHTs(), 4))
      for i in range(0, Event.GetNHTs()):
        Hits[i, 0] = Event.GetHTAt(i).GetPosition().X()
        Hits[i, 1] = Event.GetHTAt(i).GetPosition().Y()
        Hits[i, 2] = Event.GetHTAt(i).GetPosition().Z()
        Hits[i, 3] = Event.GetHTAt(i).GetEnergy()
      
      NEvents += 1
      EventTypes.append(Type)
      EventHits.append(Hits)
      
      if NEvents >= self.MaxEvents:
        break
  
    print("Occurances of different event types:")
    print(collections.Counter(EventTypes))
    
    import math

    self.LastEventIndex = 0
    self.EventHits = EventHits
    self.EventTypes = EventTypes 
    shuffledTypes = EventTypes.copy()
    shuffledHits = EventHits.copy()

    random.shuffle(shuffledHits)
    random.shuffle(shuffledTypes)
 
    ceil = math.ceil(len(self.EventHits)*0.75)
    self.EventTypesTrain = shuffledTypes[:ceil]
    self.EventTypesTest = shuffledTypes[ceil:]
    self.EventHitsTrain = shuffledHits[:ceil]
    self.EventHitsTest = shuffledHits[ceil:]

    return 


###################################################################################################


  def trainTFMethods(self):
 
    print("Starting training...")
 
    # Load the data
    #eventtypes: what we want to train {21:11, }
    #EventHits: what to conver to the point cloud
    #numpy array
    self.loadData()

    # Add VoxNet here

    print("Initializing voxnet")

    voxnet = VoxNet(self.BatchSize, self.XBins, self.YBins, self.ZBins, self.MaxLabel)
    #batch_size = 1

    p = dict() # placeholders

    p['labels'] = tf.placeholder(tf.float32, [None, self.MaxLabel])
    p['loss'] = tf.nn.softmax_cross_entropy_with_logits(logits=voxnet[-2], labels=p['labels'])
    p['loss'] = tf.reduce_mean(p['loss']) 
    p['l2_loss'] = tf.add_n([tf.nn.l2_loss(w) for w in voxnet.kernels]) 
    p['correct_prediction'] = tf.equal(tf.argmax(voxnet[-1], 1), tf.argmax(p['labels'], 1))
    p['accuracy'] = tf.reduce_mean(tf.cast(p['correct_prediction'], tf.float32))
    p['learning_rate'] = tf.placeholder(tf.float32)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      p['train'] = tf.train.AdamOptimizer(p['learning_rate'], epsilon=1e-3).minimize(p['loss'])
    p['weights_decay'] = tf.train.GradientDescentOptimizer(p['learning_rate']).minimize(p['l2_loss'])

    # Hyperparameters
    num_batches = 2147483647
    #batch_size = 50

    initial_learning_rate = 0.001
    min_learning_rate = 0.000001
    learning_rate_decay_limit = 0.0001

    #TODO://
    #not sure what supposed to go inside len
    num_batches_per_epoch = len(self.EventTypesTrain) / float(self.BatchSize)
    learning_decay = 10 * num_batches_per_epoch
    weights_decay_after = 5 * num_batches_per_epoch

    checkpoint_num = 0
    learning_step = 0
    min_loss = 1e308
    test_accuracy_baseline = 0


    print("Creating check points directory")
    if not os.path.isdir(self.Output):
      os.mkdir(self.Output)

    with open(self.Output + '/accuracies.txt', 'w') as f:
      f.write('')

    with open(self.Output + '/accuracies_labels.txt', 'w') as f:
      f.write('')

    with tf.Session() as session:
      print("Initializing global TF variables")
      session.run(tf.global_variables_initializer())
      
      for batch_index in range(num_batches):
        print("Iteration {0}".format(batch_index+1))
        
        learning_rate = max(min_learning_rate, initial_learning_rate * 0.5**(learning_step / learning_decay))
        learning_step += 1

        if batch_index > weights_decay_after and batch_index % 256 == 0:
          session.run(p['weights_decay'], feed_dict=feed_dict)

        voxs, labels = self.get_batch(self.BatchSize, True)

        tf.logging.set_verbosity(tf.logging.DEBUG)
        
        print("Starting training run")
        start = time.time()
        feed_dict = {voxnet[0]: voxs, p['labels']: labels, p['learning_rate']: learning_rate, voxnet.training: True}
        session.run(p['train'], feed_dict=feed_dict)
        print("Done with training run after {0} seconds".format(round(time.time() - start, 2)))

        if batch_index and batch_index % 8 == 0:
          print("{} batch: {}".format(datetime.datetime.now(), batch_index))
          print('learning rate: {}'.format(learning_rate))

          feed_dict[voxnet.training] = False
          loss = session.run(p['loss'], feed_dict=feed_dict)
          print('loss: {}'.format(loss))

          if (batch_index and loss > 1.5 * min_loss and learning_rate > learning_rate_decay_limit): 
            min_loss = loss
            learning_step *= 1.2
            print("decreasing learning rate...")
          min_loss = min(loss, min_loss)


        if batch_index and batch_index % 100 == 0:

          num_accuracy_batches = 30
          total_accuracy = 0
          for x in range(num_accuracy_batches):
            #TODO://
            #replace with actual data
            voxs, labels = self.get_batch(self.BatchSize, True)
            feed_dict = {voxnet[0]: voxs, p['labels']: labels, voxnet.training: False}
            total_accuracy += session.run(p['accuracy'], feed_dict=feed_dict)
          training_accuracy = total_accuracy / num_accuracy_batches
          print('training accuracy: {}'.format(training_accuracy))

          num_accuracy_batches = 90
          total_accuracy = 0
          for x in range(num_accuracy_batches):
            voxs, labels = self.get_batch(self.BatchSize, True)
            feed_dict = {voxnet[0]: voxs, p['labels']: labels, voxnet.training: False}
            total_accuracy += session.run(p['accuracy'], feed_dict=feed_dict)
          test_accuracy = total_accuracy / num_accuracy_batches
          print('test accuracy: {}'.format(test_accuracy))

          num_accuracy_batches = 90
          total_correct = []
          total_wrong = []
          for x in range(num_accuracy_batches):
            voxs, labels = self.get_batch(self.BatchSize, True)
            feed_dict = {voxnet[0]: voxs, p['labels']: labels, voxnet.training: False}
            correct_prediction = session.run(p['correct_prediction'], feed_dict=feed_dict)
            for i in range(len(correct_prediction)):
              if (correct_prediction[i] == 1):
                total_correct.append(labels[i])
              else:
                total_wrong.append(labels[i])
          sum_total_correct = sum(total_correct)
          sum_total_wrong = sum(total_wrong)
          for i in range(len(sum_total_correct)):
            if (sum_total_correct[i] == 0):
              if (sum_total_wrong[i] == 0):
                sum_total_correct[i] = 1
                sum_total_wrong[i] = -2
          test_accuracy_labels = sum_total_correct/ (sum_total_correct + sum_total_wrong)
          print('test accuracy of labels: {}'.format(test_accuracy_labels))

          # test_accuracy_labels_pos = [x for x in test_accuracy_labels if x != -1]
          # test_accuracy_baseline_pos = [x for x in test_accuracy_baseline if x != -1]

          # mean_test_accuracy = sum(test_accuracy_labels_pos)/len(test_accuracy_labels)
          # mean_accuracy_baseline = sum(test_accuracy_baseline_pos)/len(test_accuracy_baseline)

          if test_accuracy > test_accuracy_baseline:
            print('saving checkpoint {}...'.format(checkpoint_num))
            voxnet.npz_saver.save(session, self.Output + '/c-{}.npz'.format(checkpoint_num))
            with open(self.Output + '/accuracies.txt', 'a') as f:
              f.write(' '.join(map(str, (checkpoint_num, training_accuracy, test_accuracy)))+'\n')
            with open(self.Output + '/accuracies_labels.txt', 'a') as f:
              f.write(str(checkpoint_num) + " ")
              for i in test_accuracy_labels:
                f.write(str(i) + " ")
              f.write('\n')
              print('checkpoint saved!')
            test_accuracy_baseline = test_accuracy

          checkpoint_num += 1

    return
  def get_keras_model(self):
    input = tf.keras.layers.Input(batch_shape = (None, self.XBins, self.YBins, self.ZBins, 1))
    conv_1 = tf.keras.layers.Conv3D(32, 5, 2, 'valid')(input)
    batch_1 = tf.keras.layers.BatchNormalization()(conv_1)
    max_1 = tf.keras.layers.LeakyReLU(alpha = 0.1)(batch_1)

    conv_2 = tf.keras.layers.Conv3D(32, 3, 1, 'valid')(max_1)
    batch_2 = tf.keras.layers.BatchNormalization()(conv_2)
    max_2 = tf.keras.layers.LeakyReLU(alpha = 0.1)(batch_2)

    max_pool_3d = tf.keras.layers.MaxPooling3D(pool_size = (2,2,2), strides = 2)(max_2)

    reshape = tf.keras.layers.Flatten()(max_pool_3d)

    dense_1 = tf.keras.layers.Dense(64)(reshape)
    batch_5 = tf.keras.layers.BatchNormalization()(dense_1)
    activation = tf.keras.layers.ReLU()(batch_5)

    drop = tf.keras.layers.Dropout(0.2)(activation)
    dense_2 = tf.keras.layers.Dense(64)(drop)

    print("      ... output layer ...")
    output = tf.keras.layers.Softmax()(dense_2)

    model = tf.keras.models.Model(inputs = input, outputs = output)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    self.keras_model = model

    # Session configuration
    print("      ... configuration ...")
    Config = tf.ConfigProto()
    Config.gpu_options.allow_growth = True

    # Create and initialize the session
    print("      ... session ...")
    Session = tf.Session(config=Config)
    Session.run(tf.global_variables_initializer())

    print("      ... listing uninitialized variables if there are any ...")
    print(tf.report_uninitialized_variables())

    print("      ... writer ...")
    writer = tf.summary.FileWriter(self.OutputDirectory, Session.graph)
    writer.close()

    # Add ops to save and restore all the variables.
    print("      ... saver ...")
    Saver = tf.train.Saver()

    K = tf.keras.backend
    K.set_session(Session)
    return model

  def trainKerasMethods(self):
    voxnet = self.get_keras_model()
    TimeConverting = 0.0
    TimeTraining = 0.0
    TimeTesting = 0.0

    Iteration = 0
    MaxIterations = 50000
    TimesNoImprovement = 0
    MaxTimesNoImprovement = 50
    while Iteration < MaxIterations:
      Iteration += 1
      print("\n\nStarting iteration {}".format(Iteration))

      # Step 1: Loop over all training batches
      for Batch in range(0, NTrainingBatches):

        # Step 1.1: Convert the data set into the input and output tensor
        TimerConverting = time.time()

        InputTensor = np.zeros(shape=(self.BatchSize, self.XBins, self.YBins, self.ZBins, 1))
        OutputTensor = np.zeros(shape=(self.BatchSize, self.OutputDataSpaceSize))

        # Loop over all training data sets and add them to the tensor
        for g in range(0, self.BatchSize):
          Event = TrainingDataSets[g + Batch*self.BatchSize]
          # Set the layer in which the event happened
          if Event.OriginPositionZ > self.ZMin and Event.OriginPositionZ < self.ZMax:
            LayerBin = int ((Event.OriginPositionZ - self.ZMin) / ((self.ZMax- self.ZMin)/ self.ZBins) )
            OutputTensor[g][LayerBin] = 1
          else:
            OutputTensor[g][self.OutputDataSpaceSize-1] = 1

          # Set all the hit locations and energies
          for h in range(0, len(Event.X)):
            XBin = int( (Event.X[h] - self.XMin) / ((self.XMax - self.XMin) / self.XBins) )
            YBin = int( (Event.Y[h] - self.YMin) / ((self.YMax - self.YMin) / self.YBins) )
            ZBin = int( (Event.Z[h] - self.ZMin) / ((self.ZMax - self.ZMin) / self.ZBins) )
            if XBin >= 0 and YBin >= 0 and ZBin >= 0 and XBin < self.XBins and YBin < self.YBins and ZBin < self.ZBins:
              InputTensor[g][XBin][YBin][ZBin][0] = Event.E[h]

        TimeConverting += time.time() - TimerConverting

        # Step 1.2: Perform the actual training
        TimerTraining = time.time()
        #print("\nStarting training for iteration {}, batch {}/{}".format(Iteration, Batch, NTrainingBatches))
        #_, Loss = Session.run([Trainer, LossFunction], feed_dict={X: InputTensor, Y: OutputTensor})
        History = model.fit(InputTensor, OutputTensor)
        Loss = History.history['loss'][-1]
        TimeTraining += time.time() - TimerTraining

        Result = model.predict(InputTensor)

        for e in range(0, self.BatchSize):
            # Fetch real and predicted layers for training data
            real, predicted, uniqueZ = getRealAndPredictedLayers(self.OutputDataSpaceSize, OutputTensor, Result, e, Event)
            TrainingRealLayer = np.append(TrainingRealLayer, real)
            TrainingPredictedLayer = np.append(TrainingPredictedLayer, predicted)
            TrainingUniqueZLayer = np.append(TrainingUniqueZLayer, uniqueZ)

        if Interrupted == True: break

      # End for all batches

      # Step 2: Check current performance
      TimerTesting = time.time()
      print("\nCurrent loss: {}".format(Loss))
      Improvement = CheckPerformance()

      if Improvement == True:
        TimesNoImprovement = 0

        Saver.save(Session, "{}/Model_{}.ckpt".format(OutputDirectory, Iteration))

        with open(OutputDirectory + '/Progress.txt', 'a') as f:
          f.write(' '.join(map(str, (CheckPointNum, Iteration, Loss)))+'\n')

        print("\nSaved new best model and performance!")
        CheckPointNum += 1
      else:
        TimesNoImprovement += 1

      TimeTesting += time.time() - TimerTesting

      # Exit strategy
      if TimesNoImprovement == MaxTimesNoImprovement:
        print("\nNo improvement for {} iterations. Quitting!".format(MaxTimesNoImprovement))
        break;

      # Take care of Ctrl-C
      if Interrupted == True: break

      print("\n\nTotal time converting per Iteration: {} sec".format(TimeConverting/Iteration))
      print("Total time training per Iteration:   {} sec".format(TimeTraining/Iteration))
      print("Total time testing per Iteration:    {} sec".format(TimeTesting/Iteration))
        

###################################################################################################


  def get_batch(self, batch_size, train):
    """
    Main test function

    Returns
    -------
    bool
      True is everything went well, False in case of an error

    """

    rn = random.randint
    bs = batch_size
    #xmin = -55
    #ymin = -55
    #zmin = 0
    #xmax = 55
    #ymax = 55
    #zmax = 48

    if train:
      EventHits = self.EventHitsTrain
      EventTypes = self.EventTypesTrain
    else:
      EventHits = self.EventHitsTest
      EventTypes = self.EventTypesTest

    voxs = np.zeros([bs, self.XBins, self.YBins, self.ZBins, 1], dtype=np.float32)
    one_hots = np.zeros([bs, self.MaxLabel], dtype=np.float32)
    #fill event hits
    for bi in range(bs):
      self.LastEventIndex += 1
      if self.LastEventIndex == len(EventHits):
        self.LastEventIndex = 0
      while len(self.EventHitsTrain[self.LastEventIndex]) == 0:
        self.LastEventIndex += 1
        if self.LastEventIndex == len(EventHits):
          self.LastEventIndex = 0
      for i in EventHits[self.LastEventIndex]:
          xbin = (int) (((i[0] - self.XMin) / (self.XMax - self.XMin)) * self.XBins)
          ybin = (int) (((i[1] - self.YMin) / (self.YMax - self.YMin)) * self.YBins)
          zbin = (int) (((i[2] - self.ZMin) / (self.ZMax - self.ZMin)) * self.ZBins)
          #print(bi, xbin, ybin, zbin)
          voxs[bi, xbin, ybin, zbin] += i[3]
      #fills event types
      one_hots[bi][EventTypes[self.LastEventIndex]] = 1
      
    return voxs, one_hots


###################################################################################################
def getRealAndPredictedLayers(OutputDataSpaceSize, OutputTensor, Result, e, Event):
    real = 0
    predicted = 0
    unique = Event.unique
    for l in range(0, OutputDataSpaceSize):
        if OutputTensor[e][l] > 0.5:
            real = l
        if Result[e][l] > 0.5:
            predicted = l
    return real, predicted, unique

def CheckPerformance():
  global BestPercentageGood

  Improvement = False

  TotalEvents = 0
  BadEvents = 0

  # Step run all the testing batches, and detrmine the percentage of correct identifications
  # Step 1: Loop over all Testing batches
  for Batch in range(0, NTestingBatches):

    # Step 1.1: Convert the data set into the input and output tensor
    InputTensor = np.zeros(shape=(self.BatchSize, self.XBins, self.YBins, self.ZBins, 1))
    OutputTensor = np.zeros(shape=(self.BatchSize, self.OutputDataSpaceSize))


    # Loop over all testing  data sets and add them to the tensor
    for e in range(0, BatchSize):
      Event = TestingDataSets[e + Batch*BatchSize]
      # Set the layer in which the event happened
      if Event.OriginPositionZ > self.ZMin and Event.OriginPositionZ < self.ZMax:
        LayerBin = int ((Event.OriginPositionZ - self.ZMin) / ((self.ZMax- self.ZMin)/ self.ZBins) )
        #print("layer bin: {} {}".format(Event.OriginPositionZ, LayerBin))
        OutputTensor[e][LayerBin] = 1
      else:
        OutputTensor[e][self.OutputDataSpaceSize-1] = 1

      # Set all the hit locations and energies
      SomethingAdded = False
      for h in range(0, len(Event.X)):
        XBin = int( (Event.X[h] - self.XMin) / ((self.XMax - self.XMin) / self.XBins) )
        YBin = int( (Event.Y[h] - self.YMin) / ((self.YMax - self.YMin) / self.YBins) )
        ZBin = int( (Event.Z[h] - self.ZMin) / ((self.ZMax - self.ZMin) / self.ZBins) )
        #print("hit z bin: {} {}".format(Event.Z[h], ZBin))
        if XBin >= 0 and YBin >= 0 and ZBin >= 0 and XBin < self.XBins and YBin < self.YBins and ZBin < self.ZBins:
          InputTensor[e][XBin][YBin][ZBin][0] = Event.E[h]
          SomethingAdded = True

      if SomethingAdded == False:
        print("Nothing added for event {}".format(Event.ID))
        Event.print()


    # Step 2: Run it
    # Result = Session.run(Output, feed_dict={X: InputTensor})
    Result = model.predict(InputTensor)

    #print(Result[e])
    #print(OutputTensor[e])

    for e in range(0, BatchSize):
      TotalEvents += 1
      IsBad = False
      LargestValueBin = 0
      LargestValue = OutputTensor[e][0]
      for c in range(1, self.OutputDataSpaceSize) :
        if Result[e][c] > LargestValue:
          LargestValue = Result[e][c]
          LargestValueBin = c

      if OutputTensor[e][LargestValueBin] < 0.99:
        BadEvents += 1
        IsBad = True

        #if math.fabs(Result[e][c] - OutputTensor[e][c]) > 0.1:
        #  BadEvents += 1
        #  IsBad = True
        #  break

      # Fetch real and predicted layers for testing data
      real, predicted = getRealAndPredictedLayers(self.OutputDataSpaceSize, OutputTensor, Result, e)
      global TestingRealLayer
      global TestingPredictedLayer
      TestingRealLayer = np.append(TestingRealLayer, real)
      TestingPredictedLayer = np.append(TestingPredictedLayer, predicted)

      # Some debugging
      if Batch == 0 and e < 500:
        EventID = e + Batch*BatchSize + NTrainingBatches*BatchSize
        print("Event {}:".format(EventID))
        if IsBad == True:
          print("BAD")
        else:
          print("GOOD")
        DataSets[EventID].print()

        print("Results layer: {}".format(LargestValueBin))
        for l in range(0, self.OutputDataSpaceSize):
          if OutputTensor[e][l] > 0.5:
            print("Real layer: {}".format(l))
          #print(OutputTensor[e])
          #print(Result[e])

  PercentageGood = 100.0 * float(TotalEvents-BadEvents) / TotalEvents

  if PercentageGood > BestPercentageGood:
    BestPercentageGood = PercentageGood
    Improvement = True

  print("Percentage of good events: {:-6.2f}% (best so far: {:-6.2f}%)".format(PercentageGood, BestPercentageGood))

  return Improvement

###################################################################################################

  def test(self):
    """
    Main test function

    Returns
    -------
    bool
      True is everything went well, False in case of an error

    """
    # Add VoxNet here

    voxnet = VoxNet(self.BatchSize, self.XBins, self.YBins, self.ZBins, self.MaxLabel)
    #batch_size = 1

    p = dict() # placeholders

    p['labels'] = tf.placeholder(tf.float32, [200, 6697])
    p['correct_prediction'] = tf.equal(tf.argmax(voxnet[-1], 1), tf.argmax(p['labels'], 1))
    p['accuracy'] = tf.reduce_mean(tf.cast(p['correct_prediction'], tf.float32))

    num_batches = 2147483647
    #batch_size = 64

    checkpoint_num = int(max([map(float, l.split())
        for l in open('checkpoints/accuracies.txt').readlines()], 
        key=lambda x:x[2])[0])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        voxnet.npz_saver.restore(session, 'checkpoints/c-{}.npz'.format(checkpoint_num))

        total_accuracy = 0
        for batch_index in xrange(num_batches):

            voxs, labels = dataset.test.get_batch(self.BatchSize, False)
            feed_dict = {voxnet[0]: voxs, p['labels']: labels}
            total_accuracy += session.run(p['accuracy'], feed_dict=feed_dict)
            test_accuracy = total_accuracy / (batch_index+1)
            if batch_index % 32 == 0:
              print('average test accuracy: {}'.format(test_accuracy))
            return True

# END
###################################################################################################
