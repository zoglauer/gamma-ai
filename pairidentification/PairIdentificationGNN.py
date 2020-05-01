###################################################################################################
#
# PairIdentification.py
#
# Copyright (C) by Andreas Zoglauer & Harrison Costatino.
#
# Please see the file LICENSE in the main repository for the copyright-notice.
#
###################################################################################################



###################################################################################################

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np

#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

import random

import signal
import sys
import time
import math
import csv
import os
import argparse
import logging
import yaml
from datetime import datetime
from functools import reduce


print("\nPair Identification")
print("============================\n")



# Step 1: Input parameters
###################################################################################################


# Default parameters

# Split between training and testing data
TestingTrainingSplit = 0.1

MaxEvents = 1000

# File names
FileName = "PairIdentification.p1.sim.gz"
GeometryName = "$(MEGALIB)/resource/examples/geomega/GRIPS/GRIPS.geo.setup"


# Set in stone later
TestingTrainingSplit = 0.8

OutputDirectory = "Results"


parser = argparse.ArgumentParser(description='Perform training and/or testing of the pair identification machine learning tools.')
parser.add_argument('-d', '--datatype', default='tm2', help='One of: tm1: toy modle #1, tm2: toy model #2, f: file')
parser.add_argument('-f', '--filename', default='PairIdentification.p1.sim.gz', help='File name used for training/testing')
parser.add_argument('-m', '--maxevents', default='100', help='Maximum number of events to use')
parser.add_argument('-s', '--testingtrainigsplit', default='0.1', help='Testing-training split')
parser.add_argument('-b', '--batchsize', default='16', help='Batch size')

# Command line arguments for build model, to remove dependency on .yaml
parser.add_argument('--model_type', default='gnn_segment_classifier', help='model_type')
parser.add_argument('--optimizer', default='Adam', help='optimizer')
parser.add_argument('--learning_rate', default='0.001', help='learning_rate')
parser.add_argument('--loss_func', default='BCELoss', help='loss_func')
parser.add_argument('--input_dim', default='3', help='input_dim')
parser.add_argument('--hidden_dim', default='64', help='hidden_dim')
parser.add_argument('--n_iters', default='5', help='n_iters')
# parser.add_argument('--hidden_activation', default='nn.Tanh', help='hidden_activation')
parser.add_argument('--save', default='', help='save model to directory')
parser.add_argument('--restore', default='', help='restore model from file path')


args = parser.parse_args()

DataType = args.datatype

if args.filename != "":
  FileName = args.filename

if int(args.maxevents) >= 10:
  MaxEvents = int(args.maxevents)

if int(args.batchsize) >= 0:
  BatchSize = int(args.batchsize)

if float(args.testingtrainigsplit) >= 0.05:
  TestingTrainingSplit = float(args.testingtrainigsplit)


if os.path.exists(OutputDirectory):
  Now = datetime.now()
  OutputDirectory += Now.strftime("_%Y%m%d_%H%M%S")

os.makedirs(OutputDirectory)



###################################################################################################
# Step 2: Global functions
###################################################################################################


# Take care of Ctrl-C
Interrupted = False
NInterrupts = 0
def signal_handler(signal, frame):
  global Interrupted
  Interrupted = True
  global NInterrupts
  NInterrupts += 1
  if NInterrupts >= 2:
    print("Aborting!")
    sys.exit(0)
  print("You pressed Ctrl+C - waiting for graceful abort, or press  Ctrl-C again, for quick exit.")
signal.signal(signal.SIGINT, signal_handler)


# Everything ROOT related can only be loaded here otherwise it interferes with the argparse
from EventData import EventData

# Load MEGAlib into ROOT so that it is usable
import ROOT as M
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")
M.PyConfig.IgnoreCommandLineOptions = True



###################################################################################################
# Step 3: Create some training, test & verification data sets
###################################################################################################


# Read the simulation file data:
DataSets = []
NumberOfDataSets = 0

if DataType == "tm1":
  for e in range(0, MaxEvents):
    Data = EventData()
    Data.createFromToyModelRealismLevel1(e)
    DataSets.append(Data)
    
    NumberOfDataSets += 1
    if NumberOfDataSets > 0 and NumberOfDataSets % 1000 == 0:
      print("Data sets processed: {}".format(NumberOfDataSets))

elif DataType == "tm2":
  for e in range(0, MaxEvents):
    Data = EventData()
    Data.createFromToyModelRealismLevel2(e)
    DataSets.append(Data)
    
    NumberOfDataSets += 1
    if NumberOfDataSets > 0 and NumberOfDataSets % 1000 == 0:
      print("Data sets processed: {}".format(NumberOfDataSets))

elif DataType == "f":
  # Load geometry:
  Geometry = M.MDGeometryQuest()
  if Geometry.ScanSetupFile(M.MString(GeometryName)) == True:
    print("Geometry " + GeometryName + " loaded!")
  else:
    print("Unable to load geometry " + GeometryName + " - Aborting!")
    quit()


  Reader = M.MFileEventsSim(Geometry)
  if Reader.Open(M.MString(FileName)) == False:
    print("Unable to open file " + FileName + ". Aborting!")
    quit()


  print("\n\nStarted reading data sets")
  NumberOfDataSets = 0
  while NumberOfDataSets < MaxEvents:
    Event = Reader.GetNextEvent()
    if not Event:
      break

    if Event.GetNIAs() > 0:
      Data = EventData()
      if Data.parse(Event) == True:
        if Data.hasHitsOutside(XMin, XMax, YMin, YMax, ZMin, ZMax) == False:
          DataSets.append(Data)
          NumberOfDataSets += 1
          if NumberOfDataSets % 500 == 0:
            print("Data sets processed: {}".format(NumberOfDataSets))

else:
  print("Unknown data type \"{}\" Must be one of tm1, tm2, f".format(DataType))
  quit()

print("Info: Parsed {} events".format(NumberOfDataSets))

# Split the data sets in training and testing data sets

TestingTrainingSplit = 0.75


numEvents = len(DataSets)

numTraining = int(numEvents * TestingTrainingSplit)

TrainingDataSets = DataSets[:numTraining]
TestingDataSets = DataSets[numTraining:]



# For testing/validation split
# ValidationDataSets = TestingDataSets[:int(len(TestingDataSets)/2)]
# TestingDataSets = TestingDataSets[int(len(TestingDataSets)/2):]

print("###### Data Split ########")
print("Training/Testing Split: {}".format(TestingTrainingSplit))
print("Total Data: {}, Training Data: {},Testing Data: {}".format(numEvents, len(TrainingDataSets), len(TestingDataSets)))
print("##########################")


###################################################################################################
# Step 4: Vectorize data using preprocess.py
###################################################################################################

# Locals
from gnn import get_trainer
from preprocess import generate_dataset

#Externals
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

train_dataset, train_labels, train_True_Ri, train_True_Ro = generate_dataset(TrainingDataSets)
test_dataset, test_labels, test_True_Ri, test_True_Ro = generate_dataset(TestingDataSets)

train_data_loader = DataLoader(train_dataset, batch_size=BatchSize)
valid_data_loader = DataLoader(test_dataset, batch_size=BatchSize)

###################################################################################################
# Step 5: Setting up the neural network
###################################################################################################

# trainer = get_trainer(distributed=args.distributed, output_dir=output_dir,
#                           device=args.device, **experiment_config)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", "cuda:0" if torch.cuda.is_available() else "cpu", "for training.")

trainer = get_trainer(device=device)

# Build the model
# trainer.build_model(**model_config)

'''
model_config:
    model_type: 'gnn_segment_classifier'
    input_dim: 3
    hidden_dim: 64
    n_iters: 4
    loss_func: 'BCELoss'
    optimizer: 'Adam'
    learning_rate: 0.001
'''
model_type = args.model_type
optimizer = args.optimizer
learning_rate = float(args.learning_rate)
loss_func = args.loss_func
input_dim = int(args.input_dim)
hidden_dim = int(args.hidden_dim)
n_iters = int(args.n_iters)

trainer.build_model(model_type=model_type, optimizer=optimizer, learning_rate=learning_rate, loss_func=loss_func, 
  input_dim=3, hidden_dim=hidden_dim, n_iters=n_iters)

#Restore model parameters
restore_model_path = str(args.restore)
if restore_model_path:
  print('Restoring Saved Model')
  trainer.restore_model(model_path=restore_model_path)
  summary = trainer.evaluate(valid_data_loader)
  print('Loaded Model Final Valid Acc:', summary['valid_loss'][-1])

###################################################################################################
# Step 6: Training and saving the network
###################################################################################################
print("Started Training Iteration")
summary = trainer.train(train_data_loader=train_data_loader,
                        valid_data_loader=valid_data_loader, n_epochs=n_iters)
print("Finished Training")

print('Train Loss Log: ', summary['train_loss'])
print('Final Test Accuracy: ', summary['valid_acc'][-1])
print('Max Test Accuracy: ', max(summary['valid_acc']))

trainer.write_summaries("Results/result", summary)

# Save model parameters
save_model_path = str(args.save)
if save_model_path:
  print('Model Save Path:', save_model_path)
  trainer.save_model(model_path=save_model_path)

###################################################################################################
# Step 7: Evaluating and Visualizing the network
###################################################################################################

#Locals
from visualization import GraphVisualizer

viz = GraphVisualizer(summary, test_labels, test_True_Ri, test_True_Ro, OutputDirectory)
viz.plot_sample(random.randint(0, test_labels.shape[0]-1))
viz.plot_sample(random.randint(0, test_labels.shape[0]-1))
viz.plot_sample(random.randint(0, test_labels.shape[0]-1))
viz.plot_sample(random.randint(0, test_labels.shape[0]-1))
viz.plot_sample(random.randint(0, test_labels.shape[0]-1))