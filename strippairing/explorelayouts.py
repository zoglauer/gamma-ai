###################################################################################################
#
# run.py
#
# Copyright (C) by Devyn Donahou & Andreas Zoglauer.
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
from StripPairing import StripPairing
  
  
###################################################################################################


"""
This program loops over different layout and determines their performance
For all the command line options, try:

python3 explorelayouts.py --help

"""


parser = argparse.ArgumentParser(description='Perform training and/or testing of the event clustering machine learning tools.')
parser.add_argument('-f', '--file', default='EC.hits4.groups3.eventclusterizer.root', help='File name used for training/testing')
#parser.add_argument('-c', '--complete', action='store_true', help='Try to find similar data files and train/test them too')
parser.add_argument('-o', '--output', default='Results', help='Prefix for the output filename and directory')
#parser.add_argument('-b', '--energy', default='0,10000', help='Energy bins. Example: 0,10000')
parser.add_argument('-l', '--hiddenlayers', default='2', help='Number of hidden layers. Default: 2')
parser.add_argument('-n', '--maximumnodes', default='50', help='Maximum number of nodes per hidden layer. Default: 50')
#parser.add_argument('-a', '--algorithm', default='TMVA:BDT', help='Machine learning algorithm. Allowed: TMVA:MLP')
parser.add_argument('-m', '--maxevents', default='100000', help='Maximum number of events to use')
#parser.add_argument('-e', '--onlyevaluate', action='store_true', help='Only test the approach')
#parser.add_argument('-t', '--onlytrain', action='store_true', help='Only train the approach')

args = parser.parse_args()

# Step 1: Create list all layouts

LayoutList = []
for X in list(itertools.product(range(5, int(args.maximumnodes)+1, 5), repeat=int(args.hiddenlayers))):
  Layout = ""
  for e in X:
    if Layout != "":
      Layout += ","
    Layout += str(e)
  LayoutList.append(Layout)
  print(Layout)


# Step 2: Loop over all layout and record performance 

for Layout in LayoutList:
  AI = StripPairing(args.file, args.output, Layout, int(args.maxevents))

  #if AI.train() == False:
    #continue

  #Passed, PerformanceGoodSequences, PerformanceBadSequence = AI.test()
  
  #if Passed == True:
    # Store Performances in List

# Step 3: Make nice performance graphs



# END
###################################################################################################
