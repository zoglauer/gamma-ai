"""Contains EventData class to store simulation data in standardized format.

EventData: main class.
__init__(self): constructor for EventData.
print(self): prints contents of instance of EventData.

Used in event_extractor.py, which does the actual processing of simulation
data. Resulting processed data, consisting of instances of EventData, is stored
to a file that is later imported to energy_estimate.py as training/testing
data for neural network models in that file.
"""

import numpy as np

class EventData:
    """
    Stores data of one detected event.

    __init__(self): constructor.

    print(self): prints data associated with event.
    """

    def __init__(self):
        """
        Default constructor.

        id_: unique identifier for event.
        hits: list of interactions with their energies and coordinates.
        type: 1: Compton, 2: pair.
        detector: 0: passive, 1: tracker, 2: absorber.
        measured_energy: energy measured by detector.
        gamma_energy: 'true' total energy of interaction. Larger than measured.
        """
        self.id_ = 0

        self.hits = np.array([])

        self.type = 0
        self.detector = 0

        self.measured_energy = 0.0
        self.gamma_energy = 0.0

    def print(self):
        """Prints data associated with event passed in."""

        print(f"Event ID: {self.id_}")
        print(f"  Type: {self.type}  Detector: {self.detector}")
        print(f"  Hits: {self.hits}")
        print(f"  Measured Energy: {self.measured_energy}  Gamma Energy: {self.gamma_energy}")