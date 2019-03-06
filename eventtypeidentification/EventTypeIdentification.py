###################################################################################################
#
# EventTypeIdentification.py
#
# Copyright (C) by Andreas Zoglauer, Amal Metha & Caitlyn Chen.
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
    self.OutputPrefix = Output
    self.Algorithms = Algorithm
    self.MaxEvents = MaxEvents

    self.EventTypes = []
    self.EventHits = []
    self.LastEventIndex = 0


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
          Type += 10 + Event.GetIAAt(1).GetDetectorType()
        elif Event.GetIAAt(1).GetProcess() == M.MString("PAIR"):
          Type += 20 + Event.GetIAAt(1).GetDetectorType()
      else:
        break  
  
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
    
    self.LastEventIndex = 0
    self.EventHits = EventHits
    self.EventTypes = EventTypes  

    return 

###################################################################################################


  def trainTFMethods(self):
  
    # Load the data
    #eventtypes: what we want to train {21:11, }
    #EventHits: what to conver to the point cloud
    #numpy array
    self.loadData()

    # Add VoxNet here
    #dataset = getBatch(1, __batchsize_)
    voxnet = VoxNet()
    batch_size = 1

    p = dict() # placeholders

    p['labels'] = tf.placeholder(tf.float32, [None, 6697])
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
    batch_size = 200

    initial_learning_rate = 0.001
    min_learning_rate = 0.000001
    learning_rate_decay_limit = 0.0001

    #TODO://
    #not sure what supposed to go inside len
    num_batches_per_epoch = len(self.EventTypes) / float(batch_size)
    learning_decay = 10 * num_batches_per_epoch
    weights_decay_after = 5 * num_batches_per_epoch

    checkpoint_num = 0
    learning_step = 0
    min_loss = 1e308

    if not os.path.isdir('checkpoints'):
      os.mkdir('checkpoints')

    with open('checkpoints/accuracies.txt', 'w') as f:
      f.write('')

    with tf.Session() as session:
      session.run(tf.global_variables_initializer())

      for batch_index in range(num_batches):
        learning_rate = max(min_learning_rate, initial_learning_rate * 0.5**(learning_step / learning_decay))
        learning_step += 1

        if batch_index > weights_decay_after and batch_index % 256 == 0:
          session.run(p['weights_decay'], feed_dict=feed_dict)

        voxs, labels = self.get_batch(batch_size)

        
        feed_dict = {voxnet[0]: voxs, p['labels']: labels, p['learning_rate']: learning_rate, voxnet.training: True}
        session.run(p['train'], feed_dict=feed_dict)

        if batch_index and batch_index % 512 == 0:
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

        if batch_index and batch_index % 2048 == 0:
          num_accuracy_batches = 30
          total_accuracy = 0
          for x in range(num_accuracy_batches):
            #TODO://
            #replace with actual data
            voxs, labels = self.get_batch(batch_size)
            feed_dict = {voxnet[0]: voxs, p['labels']: labels, voxnet.training: False}
            total_accuracy += session.run(p['accuracy'], feed_dict=feed_dict)
          training_accuracy = total_accuracy / num_accuracy_batches
          print('training accuracy: {}'.format(training_accuracy))

          num_accuracy_batches = 90
          total_accuracy = 0
          for x in range(num_accuracy_batches):
            voxs, labels = self.get_batch(batch_size)
            feed_dict = {voxnet[0]: voxs, p['labels']: labels, voxnet.training: False}
            total_accuracy += session.run(p['accuracy'], feed_dict=feed_dict)
          test_accuracy = total_accuracy / num_accuracy_batches
          print('test accuracy: {}'.format(test_accuracy))

          print('saving checkpoint {}...'.format(checkpoint_num))
          voxnet.npz_saver.save(session, 'checkpoints/c-{}.npz'.format(checkpoint_num))
          with open('checkpoints/accuracies.txt', 'a') as f:
            f.write(' '.join(map(str, (checkpoint_num, training_accuracy, test_accuracy)))+'\n')
          print('checkpoint saved!')

          checkpoint_num += 1

    return


  def get_batch(self, batch_size):
    rn = random.randint
    bs = batch_size
    xmin = -55
    ymin = -55
    zmin = 0
    xmax = 55
    ymax = 55
    zmax = 48
    xbins = 110
    ybins = 110
    zbins = 48
    voxs = np.zeros([bs, xbins,ybins,zbins, 1], dtype=np.float32)
    one_hots = np.zeros([bs, len(self.EventTypes)], dtype=np.float32)
    #fill event hits
    for bi in range(bs):
      self.LastEventIndex += 1
      if self.LastEventIndex == len(self.EventHits):
        self.LastEventIndex = 0
      while len(self.EventHits[self.LastEventIndex]) == 0:
        self.LastEventIndex += 1
        if self.LastEventIndex == len(self.EventHits):
          self.LastEventIndex = 0
      for i in self.EventHits[self.LastEventIndex]:
          xbin = (int) (((i[0] - xmin) / (xmax - xmin)) * xbins)
          ybin = (int) (((i[1] - ymin) / (ymax - ymin)) * ybins)
          zbin = (int) (((i[2] - zmin) / (zmax - zmin)) * zbins)
          #print(bi, xbin, ybin, zbin)
          voxs[bi, xbin, ybin, zbin] += i[3]
      #fills event types
      one_hots[bi][self.EventTypes[self.LastEventIndex]] = 1
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

    return True




# END
###################################################################################################
