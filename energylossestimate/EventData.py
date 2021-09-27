import numpy as np

class EventData:
    """
    stores data of one event.
    type: 1: Compton, 2: pair.
    detector: 0: passive, 1: tracker, 2: absorber
    position: x, y, z
    energy: energy, GammaEnergy
    """

    def __init__(self):
        """
        default constructor.
        creates ID, Hits, type, detector, startX, startY, startZ,
        energy, and GammaEnergy.
        """
        # may need to reformat type/detector for NN
        self.ID = 0

        self.Hits = np.array([])

        self.type = 0
        self.detector = 0

        #self.startX = np.array([])
        #self.startY = np.array([])
        #self.startZ = np.array([])

        self.measured_energy = 0.0
        self.GammaEnergy = 0.0

    #any other funcs needed like in recoil electron?

    def print(self):#may need updating if __init__ changes
        """
        Print data
        """

        print("Event ID: {}".format(self.ID))
        #print("  Start: {} {} {}".format(self.startX, self.startY, self.startZ))
        print("  Type: {}  Detector: {}".format(self.type, self.detector))
        print("  Hits: {}".format(self.Hits))
        print("  Energy: {}  Gamma Energy: {}".format(self.Energy, self.GammaEnergy))
