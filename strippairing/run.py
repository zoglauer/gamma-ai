import argparse
from strippairing3 import StripPairing

parser = argparse.ArgumentParser(description='Perform training and/or testing of the strip pairing machine learning tools.')
parser.add_argument('-f', '--file', default='StripPairing.x2.y2.strippairing.root', help='File name used for training/testing')
parser.add_argument('-o', '--output', default='Results', help='Prefix for the output filename and directory')
parser.add_argument('-l', '--layout', default='N+5,N', help='Layout of the hidden layer')
parser.add_argument('-e', '--onlyevaluate', action='store_true', help='Only test the approach')

args = parser.parse_args()

SP = StripPairing(args.file, args.output, args.layout)
if args.onlyevaluate == False:
  SP.train()
SP.test()

