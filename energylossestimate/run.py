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
import keras
from EnergyLossEstimate import *

###################################################################################################


"""
This is the main program for the energy loss identification testing and training in python.
For all the command line options, try:

python3 run.py --help

"""

print("Starting event type identification")

parser = argparse.ArgumentParser(
    description='Perform training and/or testing of the energy loss estimate machine learning tools.')
parser.add_argument('-f', '--file', default='/volumes/selene/users/rithwik/2MeV_5GeV_flat.inc1.id1.sim.gz',
                    help='File name used for training/testing')
parser.add_argument('-o', '--output', default='',
                    help='Postfix for the output directory')
parser.add_argument('-a', '--algorithm', default='median',
                    help='Machine learning algorithm. Allowed: TF:VOXNET')
parser.add_argument('-m', '--maxevents', default='100000',
                    help='Maximum number of events to use')
parser.add_argument('-e', '--onlyevaluate', default="False",
                    help='Only test the approach')
parser.add_argument('-p', '--onlyplots', default="False",
                    help='Only save 2D histogram / scatterplot of gamma vs. detected energies')


args = parser.parse_args()

AI = keras.EnergyLossEstimate(
    args.file, args.output, args.algorithm, int(args.maxevents))

if args.onlyplots == "True":
    medianModel = medianModel(AI)
    medianModel.plotHist()
    medianModel.plotMedians()
    # AI.plotScatter()
    sys.exit()

if args.onlyevaluate == "False":
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
