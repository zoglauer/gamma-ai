###################################################################################################
#
# optimizerBDT.py
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
from EnergyLoss import EnergyLossIdentification
  
  
###################################################################################################


"""
This is the main program for the energy loss identification testing and training in python.
For all the command line options, try:

python3 optimizerBDT.py --help

"""


parser = argparse.ArgumentParser(description='Optimize the BDT energy loss identifier.')
parser.add_argument('-f', '--file', default='EC.hits4.groups3.eventclusterizer.root', help='File name used for training/testing')
parser.add_argument('-m', '--maxevents', default='100000', help='Maximum number of events to use')

args = parser.parse_args()

NTrees = [ 100, 200, 500, 1000, 2000, 5000, 10000 ]
BestNTrees = 0;
BestROC = 0

for Trees in NTrees:
  AI = EnergyLossIdentification(args.file, "Result", "TMVA:BDT", int(args.maxevents))
  AI.setBDTValues(Trees, 1, 3, 0.4)
  AI.train()
  Results = AI.getTMVAResults()
  if Results["BDT"] > BestROC:
    BestNTrees = Trees
    BestROC = Results["BDT"]
    
    
print("Best ROC {} for {} trees".format(BestROC, BestNTrees))

# prevent Canvases from closing

List = ROOT.gROOT.GetListOfCanvases()
if List.LastIndex() > 0:
  print("ATTENTION: Please exit by clicking: File -> Close ROOT! Do not just close the window by clicking \"x\"")
  print("           ... and if you didn't honor this warning, and are stuck, execute the following in a new terminal: kill " + str(os.getpid()))
  ROOT.gApplication.Run()


# END
###################################################################################################
