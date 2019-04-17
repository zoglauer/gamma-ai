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


###################################################################################################


  def train(self):
    """
    Switch between the various machine-learning libraries based on self.Algorithm
    """

    if self.Algorithms.startswith("TF:"):
      self.trainTFMethods()
    #elif self.Algorithms.startswith("TMVA:"):
    #  self.trainTMVAMethods()
    #elif self.Algorithms.startswith("SKL:"):
    #  self.trainSKLMethods()
    else:
      print("ERROR: Unknown algorithm: {}".format(self.Algorithms))

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
          total_accuracy = 0
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
          print(sum(total_correct))
          print(sum(total_wrong))
          test_accuracy = sum(total_correct) / (sum(total_correct) + sum(total_wrong))
          print('test accuracy: {}'.format(test_accuracy))

          if test_accuracy > test_accuracy_baseline:
            print('saving checkpoint {}...'.format(checkpoint_num))
            voxnet.npz_saver.save(session, self.Output + '/c-{}.npz'.format(checkpoint_num))
            with open(self.Output + '/accuracies.txt', 'a') as f:
              f.write(' '.join(map(str, (checkpoint_num, training_accuracy, test_accuracy)))+'\n')
              print('checkpoint saved!')
            test_accuracy_baseline = test_accuracy

          checkpoint_num += 1

    return


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
    xmin = -55
    ymin = -55
    zmin = 0
    xmax = 55
    ymax = 55
    zmax = 48

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
          xbin = (int) (((i[0] - xmin) / (xmax - xmin)) * self.XBins)
          ybin = (int) (((i[1] - ymin) / (ymax - ymin)) * self.YBins)
          zbin = (int) (((i[2] - zmin) / (zmax - zmin)) * self.ZBins)
          #print(bi, xbin, ybin, zbin)
          voxs[bi, xbin, ybin, zbin] += i[3]
      #fills event types
      one_hots[bi][EventTypes[self.LastEventIndex]] = 1
      
    return voxs, one_hots


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
