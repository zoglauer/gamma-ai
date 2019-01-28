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
import numpy as np
from StripPairing import StripPairing
import matplotlib.pyplot as plt
  
  
###################################################################################################


"""
This program loops over different layout and determines their performance
For all the command line options, try:

python3 explorelayouts.py --help

"""


parser = argparse.ArgumentParser(description='Perform training and/or testing of the event clustering machine learning tools.')
parser.add_argument('-f', '--file', default='StripPairing.x2.y2.strippairing.root', help='File name used for training/testing')
#parser.add_argument('-c', '--complete', action='store_true', help='Try to find similar data files and train/test them too')
parser.add_argument('-o', '--output', default='Results', help='Prefix for the output filename and directory')
parser.add_argument('-l', '--hiddenlayers', default='2', help='Number of hidden layers. Default: 2')
parser.add_argument('-n', '--maximumnodes', default='50', help='Maximum number of nodes per hidden layer. Default: 50')
#parser.add_argument('-a', '--algorithm', default='TMVA:BDT', help='Machine learning algorithm. Allowed: TMVA:MLP')
parser.add_argument('-m', '--maxevents', default='100000', help='Maximum number of events to use')
#parser.add_argument('-e', '--onlyevaluate', action='store_true', help='Only test the approach')
#parser.add_argument('-t', '--onlytrain', action='store_true', help='Only train the approach')
parser.add_argument('-b', '--batch', action='store_true', help='Batch mode - don\'t show any histograms')

args = parser.parse_args()

# Step 1: Create list all layouts

LayoutList = []
for X in list(itertools.product(range(15, int(args.maximumnodes)+1, 5), repeat=int(args.hiddenlayers))):
  Layout = ""
  for e in X:
    if Layout != "":
      Layout += ","
    Layout += str(e)
  LayoutList.append(Layout)
  print(Layout)


# Step 2: Start saving the raw data to file:
f = open(args.output + ".results",'w')
f.write("# Hidden layers: {0}\n".format(args.hiddenlayers))
f.write("# maximum nodes: {0}\n".format(args.maximumnodes))
f.write("\n")


# Step 3: Loop over all layout and record performance 
GoodSequences = []
BadSequences = []
# for Layout in LayoutList:
for i in range(0, len(LayoutList)):
  AI = StripPairing(args.file, args.output, LayoutList[i], int(args.maxevents))

  if AI.train() == False:
    continue
  
  Passed, PerformanceGoodSequences, PerformanceBadSequence = AI.test()

  #Passed = True
  #PerformanceGoodSequences = i*100
  #PerformanceBadSequence = 100

  f.write(LayoutList[i] + " ")

  if Passed == True:
    # Store Performances in List
    GoodSequences.append(PerformanceGoodSequences)
    BadSequences.append(PerformanceBadSequence)
    f.write(" {0} {1}\n".format(PerformanceGoodSequences, PerformanceBadSequence))
  else:
    GoodSequences.append(0)
    BadSequences.append(1)
    f.write(" {0} {1}\n".format(0.0, 0.0))

  f.flush()
    
f.close()


# Step 4: Make nice performance graphs

if args.batch == False:
  # Simple histogram
  if int(args.hiddenlayers) == 1:

    print([x/y for x, y in zip(GoodSequences, BadSequences)])
    print(range(15, int(args.maximumnodes)+1, 5))
    
    plt.plot(range(15, int(args.maximumnodes)+1, 5), [x/y for x, y in zip(GoodSequences, BadSequences)])

    plt.title("Ratio Of Good to Bad Sequences", fontsize=20)
    plt.xlabel('Nodes in Hidden Layer', fontsize=16)
    plt.ylabel('Ratio of Good to Bad', fontsize=16)
  
    plt.show()
  
  elif int(args.hiddenlayers) == 2:
    x = range(15, int(args.maximumnodes)+1, 5)
    y = range(15, int(args.maximumnodes)+1, 5)
    xm, ym = np.meshgrid(x,y)
    
    z = [x/y for x, y in zip(GoodSequences, BadSequences)]
    z = np.array(z)
    z = z.reshape(len(xm), len(ym))
    
    fig = plt.figure();
    ax = fig.gca()
    
    #res = ax.contourf(xm,ym,z)
    #fig.colorbar(res)
    #ax.set_title('contourf')
    
    res = ax.matshow(z, origin = 'lower', aspect = 'auto', extent=[15-2.5, float(args.maximumnodes)+2.5, 15-2.5, float(args.maximumnodes)+2.5],)
    fig.colorbar(res)
    ax.set_title('matshow', y=1.1)

    
    plt.show()
    
  else:
    print("Plotting for more than one hidden layer not yet implemented")
  


print(GoodSequences)
print(BadSequences)

# END
###################################################################################################
