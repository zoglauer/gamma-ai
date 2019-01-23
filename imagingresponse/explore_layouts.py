###################################################################################################
#
#
# Copyright (C) by Shivani Kishnani & Andreas Zoglauer.
# All rights reserved.
#
# Please see the file License.txt in the main repository for the copyright-notice. 
#  
###################################################################################################

  
  
###################################################################################################


import os
import sys
import argparse
import itertools
import ROOT
from ToyModel3DCone import ToyModel3DCone
  
  
###################################################################################################


"""
This program loops over different layout and determines their performance
For all the command line options, try:

python3 explorelayouts.py --help

"""

parser = argparse.ArgumentParser(description='Perform training and/or testing of the event clustering machine learning tools.')
parser.add_argument('-f', '--file', default='changethis', help='File name used for training/testing')
parser.add_argument('-o', '--output', default='output.txt', help='The output file name where the final results will be stored')
parser.add_argument('-l', '--hiddenlayers', default='3', help='Number of hidden layers. Default: 3')
parser.add_argument('-n', '--startingnode', default='50', help='Maximum number of nodes per hidden layer. Default: 50')
parser.add_argument('-m', '--multfactor', default='10', help='Number that is to be multiplied to starting nodes to get layers of new file')
parser.add_argument('-a', '--activation', default='relu', help='Name of default activation layer to be applied')

args = parser.parse_args()

hiddenLayers = int(args.hiddenlayers)
multFactor = int(args.multfactor)
startingNode = int(args.startingnode)

# Step 1: Create function to get layout
def create_layout(node, numlayers):
	if numLayers > 0:
		return [node].extend(create_layout(node*multFactor, numLayers-1))
	return []

# Step 2: Create list of layouts for NN
LayoutList = []
for layer in create_layout(startingNode, hiddenlayers):
  Layout = ""
  for indNode in layer:
    if Layout != "":
      Layout += ","
    Layout += str(indNode)
  LayoutList.append(Layout)
  print(Layout)

# Step 3: Loop over all layout and record performance 

#for Layout in LayoutList:
  #AI = ToyModel3DCone(args.file, args.output, arg.hiddenlayers, int(args.maximumnodes))

  #if AI.train() == False:
   # continue

  #Passed, PerformanceGoodSequences, PerformanceBadSequence = AI.test()
  
  #if Passed == True:
    # Store Performances in List

# Step 3: Make nice performance graphs



# END
###################################################################################################
