import ROOT
import array
import sys
from BASE import BASE

class CERA(BASE):
    def __init__(self, filename, quality):
        sigCut = ROOT.TCut("EvaluationIsReconstructable >= 0.5")
        bgCut = ROOT.TCut("EvaluationIsReconstructable < 0.5")
        BASE.__init__(self, filename, quality, sigCut, bgCut)

    #@Override
    def eval(self):
        branches = self.branches
        variablemap = self.variablemap
        datatree = self.datatree
        reader = self.reader

        for b in list(branches):
          if b.GetName().startswith("EvaluationIsReconstructable") or b.GetName().startswith("EvaluationIsCompletelyAborbed"):
            variablemap[b.GetName()] = array.array('f', [0])
            datatree.SetBranchAddress(b.GetName(), variablemap[b.GetName()])

        reader.BookMVA("BDT","Results/weights/TMVAClassification_BDT.weights.xml")

        NEvents = 0
        NGoodEvents = 0

        NLearnedGoodEvents = 0
        NLearnedCorrectEvents = 0

        varx = array.array('f',[0])
        vary = array.array('f',[0])

        for x in range(0, min(500, datatree.GetEntries())):
          datatree.GetEntry(x)

          NEvents += 1

          print("\nSimulation ID: " + str(int(variablemap["SimulationID"][0])) + ":")

          result = reader.EvaluateMVA("BDT")
          vary.append(result)

          r = 2
          IsGood = True
          IsGoodThreshold = 0.2

          IsLearnedGood = True
          IsLearnedGoodThreshold = 0.06 # Adjust this as see fit

          for b in list(branches):
            name = b.GetName()

            if name.startswith("EvaluationIsReconstructable") or name.startswith("EvaluationIsCompletelyAborbed"):
              print(name + " " + str(variablemap[name][0]))
              if not variablemap[name][0]:
                IsGood = False
              if result < IsLearnedGoodThreshold:
                IsLearnedGood = False
              r += 1

          if IsGood == True:
            NGoodEvents += 1
            print(" --> Good event")
          else:
            print(" --> Bad event")

          if (IsLearnedGood == True and IsGood == True) or (IsLearnedGood == False and IsGood == False):
            NLearnedCorrectEvents += 1

        print("\nResult:")
        print("All events: " + str(NEvents))
        print("Good events: " + str(NGoodEvents))
        print("Correctly identified: " + str(NLearnedCorrectEvents / NEvents))

        return varx, vary
