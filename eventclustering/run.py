import os
import argparse
from EC import EventClustering
import ROOT

parser = argparse.ArgumentParser(description='Perform training and/or testing of the event clustering machine learning tools.')
parser.add_argument('-f', '--file', default='EC.maxhits3.eventclusterizer.root', help='File name used for training/testing')
parser.add_argument('-o', '--output', default='Results', help='Prefix for the output filename and directory')
parser.add_argument('-l', '--layout', default='2*N,N', help='Layout of the hidden layer')
parser.add_argument('-a', '--algorithm', default='MLP', help='Machine learning algorithm. Allowed: MLP')
parser.add_argument('-e', '--onlyevaluate', action='store_true', help='Only test the approach')

args = parser.parse_args()

AI = EventClustering(args.file, args.output, args.layout, args.algorithm)
if args.onlyevaluate == False:
  AI.train()
AI.test()


# prevent Canvases from closing

List = ROOT.gROOT.GetListOfCanvases()
if List.LastIndex() > 0:
  print("ATTENTION: Please exit by clicking: File -> Close ROOT! Do not just close the window by clicking \"x\"")
  print("           ... and if you didn't honor this warning, and are stuck, execute the following in a new terminal: kill " + str(os.getpid()))
  ROOT.gApplication.Run()