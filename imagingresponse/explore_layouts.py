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
from ToyModel3DCone import ToyModel3DCone
import signal
  
###################################################################################################


"""
This program loops over different layout and determines their performance
For all the command line options, try:

python3 explorelayouts.py --help

"""

parser = argparse.ArgumentParser(description='Passing in values to run ToyModel3DCone to test different layouts')
parser.add_argument('-f', '--file', default='changethis.txt', help='File name used for training/testing')
parser.add_argument('-o', '--output', default='output.txt', help='The output file name where the final results will be stored')
parser.add_argument('-l', '--hiddenlayers', default='3', help='Number of hidden layers. Default: 3')
parser.add_argument('-n', '--startingnode', default='10', help='Number of nodes to start with. Default: 50')
parser.add_argument('-m', '--multfactor', default='10', help='Number that is to be multiplied to starting nodes to get layers of new file')
parser.add_argument('-a', '--activation', default='relu', help='Name of default activation layer to be applied')
parser.add_argument('-mn', '--maxNode', default='50', help='Maximum number of nodes in a layer')
parser.add_argument('-t', '--time', default='600', help='Time in seconds to run the model for')

args = parser.parse_args()

hiddenLayers = int(args.hiddenlayers)
multFactor = int(args.multfactor)
startingNode = int(args.startingnode)
maxNode = int(args.maxNode)
LayoutList = []
output = args.output
filew = open(output,"w+")

#Step 0: Take care of Ctrl+C
Interrupted = False
NInterrupts = 0

def signal_handler(signal, frame):
      print("You pressed Ctrl+C! inside explore_layouts!")
      global Interrupted
      Interrupted = True        
      global NInterrupts
      NInterrupts += 1
      if NInterrupts >= 3:
        print("Aborting!")
        filew.close()
        System.exit(0)
      signal.signal(signal.SIGINT, signal_handler)

# Step 1: Create function to get layout
def create_layout(node, numLayers):
	layer_list = [node]
	while numLayers > 0 and node!= 0:
		add = node*multFactor
		
		layer_list.append(node*multFactor)
		node = add
		numLayers -= 1
	return layer_list

# Step 2: Create list of layouts for NN

for Layout in list(create_layout(x, hiddenLayers) for x in range(startingNode, maxNode+1, 10)): 
	LayoutList.append(Layout)
	print(Layout)


# Step 3: Loop over all layouts and record performance 

for Layout in LayoutList:
	ToyModel3DCone(filew, Layout, args.activation)

filew.close()
print("Finished!")

# END
###################################################################################################
