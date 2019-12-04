###################################################################################################
#
# run.py
#
# Copyright (C) by Andreas Zoglauer.
# All rights reserved.
#
# Please see the file License.txt in the main repository for the copyright-notice. 
#  
###################################################################################################

  
  
###################################################################################################


import os
import sys
import argparse
import ROOT
from EventTypeIdentification import EventTypeIdentification
  
  
###################################################################################################


"""
This is the main program for the energy loss identification testing and training in python.
For all the command line options, try:

python3 run.py --help

"""

print("Starting event type identification")

parser = argparse.ArgumentParser(description='Perform training and/or testing of the event type identification machine learning tools.')
parser.add_argument('-f', '--file', default='EC.hits4.groups3.eventclusterizer.root', help='File name used for training/testing')
parser.add_argument('-o', '--output', default='', help='Postfix for the output directory')
parser.add_argument('-a', '--algorithm', default='KERAS:VOXNET', help='Machine learning algorithm. Allowed: TF:VOXNET')
parser.add_argument('-m', '--maxevents', default='100000', help='Maximum number of events to use')
parser.add_argument('-e', '--onlyevaluate', action='store_true', help='Only test the approach')

args = parser.parse_args()

AI = EventTypeIdentification(args.file, args.output, args.algorithm, int(args.maxevents))

if args.onlyevaluate == False:
  if AI.train() == False:
    sys.exit()

if AI.test() == False:
  sys.exit()


# prevent Canvases from closing

List = ROOT.gROOT.GetListOfCanvases()
if List.LastIndex() > 0:
  print("ATTENTION: Please exit by clicking: File -> Close ROOT! Do not just close the window by clicking \"x\"")
  print("           ... and if you didn't honor this warning, and are stuck, execute the following in a new terminal: kill " + str(os.getpid()))
  ROOT.gApplication.Run()


# END
###################################################################################################
