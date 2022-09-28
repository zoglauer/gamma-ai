"""Processes data from simulation file into EventData class format.

Dependencies: signal, sys, pickle, argparse, numpy, ROOT, event_data

Parameters taken:
 -f: specifies filename to pull simulation data from.
 -m: specifies maximum number of data sets to extract.
 -d: enables debugging output.
 -p: selects parser function appropriate for algorithm taking data.

Functions:
 parse_voxnet(sim_event, debug=None): parses data into EventData as appropriate
  for voxnet neural networks.
 signal_handler(signal, frame): handles Ctrl-C interrupts.
"""

import signal
import sys
import pickle
import argparse
import numpy as np
import ROOT as M

from EventData import EventData

print("f{time.time()}: Load data from sim file")

# Load MEGAlib into ROOT:

M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")

# Initialize MEGAlib:

G = M.MGlobal()
G.Initialize()

# Handle input arguments:

parser = argparse.ArgumentParser(
    description='Extract events from Cosima sim file.')
parser.add_argument('-f', '--filename', default='EnergyEstimate.p1.sim.gz',
                    help='File name used for training/testing')
parser.add_argument('-m', '--maxdatasets', default=1000000000, type=int,
                    help='Maximum number of good data sets to extract')
parser.add_argument('-d', '--debug', default=False, action="store_true",
                    help='Enable debugging output')
parser.add_argument('-p', '--parser', default='voxnet',
                    help='Choose parser appropriate for algorithm taking data.')

args = parser.parse_args()

file_name = args.filename

max_data_sets = args.maxdatasets

output_file_name = file_name
if output_file_name.endswith(".gz"):
    output_file_name = output_file_name[:-3]
if output_file_name.endswith(".sim"):
    output_file_name = output_file_name[:-4]
output_file_name += ".data"

debug = args.debug

# Parser functions:


def parse_voxnet(sim_event, debug=None):
    """Parses an individual event into EventData class for voxnet models.

    Extracts data from sim_event to prepare for Keras voxnet models in
    energy_loss_estimate.py. Formats using EventData() class from
    event_data.py.

    When debug is set to True, will print info helpful for debugging.
    """

    data = EventData()
    data.id_ = sim_event.GetID()

    # Set type, detector

    if sim_event.GetNIAs() > 0:  # Check if interactions present
        if sim_event.GetIAAt(1).GetProcess() == M.MString("COMP"):
            data.type = 0
        if sim_event.GetIAAt(1).GetProcess() == M.MString("PAIR"):
            data.type = 1

        data.detector = sim_event.GetIAAt(1).GetDetectorType()
        data.gamma_energy = sim_event.GetIAAt(0).GetSecondaryEnergy()
    elif debug:
        print("No interactions present.")

    # Calculate total measured energy, create list of "hits"

    total_measured_energy = 0
    hits = np.zeros((sim_event.GetNHTs(), 4))
    for i in range(0, sim_event.GetNHTs()):
        x_pos = sim_event.GetHTAt(i).GetPosition().X()
        y_pos = sim_event.GetHTAt(i).GetPosition().Y()
        z_pos = sim_event.GetHTAt(i).GetPosition().Z()
        hits[i, 0] = x_pos
        hits[i, 1] = y_pos
        hits[i, 2] = z_pos
        hit_energy = sim_event.GetHTAt(i).GetEnergy()
        hits[i, 3] = hit_energy
        total_measured_energy += hit_energy

    data.hits = hits
    data.measured_energy = total_measured_energy

    if debug:
        print(sim_event.ToSimString().Data())
        data.print()

    return data

# Select parser to use based upon arguments:


parser_options = {'voxnet': parse_voxnet}
parser_to_use = parser_options[args.parser]

# Load geometry:

# Geometry to use. Fixed for the time being
GEOMETRY_NAME = "$(MEGALIB)/resource/examples/geomega/GRIPS/GRIPS_extended.geo.setup"

geometry = M.MDGeometryQuest()
if geometry.ScanSetupFile(M.MString(GEOMETRY_NAME)):
    print("Geometry " + GEOMETRY_NAME + " loaded!")
else:
    print("Unable to load geometry " + GEOMETRY_NAME + " - Aborting!")
    sys.exit()

reader = M.MFileEventsSim(geometry)
if not reader.Open(M.MString(file_name)):
    print("Unable to open file " + file_name + ". Aborting!")
    sys.exit()

# Handle Ctrl-C:

INTERRUPTED = False
N_INTERRUPTS = 0


def signal_handler(signal, frame):
    """Handles Ctrl-C interrupts."""
    global INTERRUPTED
    global N_INTERRUPTS
    INTERRUPTED = True
    N_INTERRUPTS += 1
    if N_INTERRUPTS >= 2:
        print("Aborting!")
        sys.exit(0)
    print("You pressed Ctrl+C - wait for graceful abort, or press Ctrl-C again for quick exit.")


signal.signal(signal.SIGINT, signal_handler)

# Loop for actual data processing:

data_sets = []

NUMBER_OF_EVENTS = 0
NUMBER_OF_DATA_SETS = 0

print("\n\nStarted reading data sets")
while True:
    event = reader.GetNextEvent()

    if not event:
        break
    # Python needs ownership of the event in order to delete it
    M.SetOwnership(event, True)

    NUMBER_OF_EVENTS += 1
    parsed_data = parser_to_use(event, debug)

    if parsed_data is not None:
        data_sets.append(parsed_data)
        NUMBER_OF_DATA_SETS += 1

    if NUMBER_OF_DATA_SETS > 0 and NUMBER_OF_EVENTS % 1000 == 0:
        print(
            f"Data sets processed: {NUMBER_OF_DATA_SETS} / {NUMBER_OF_EVENTS}")

    if NUMBER_OF_DATA_SETS >= max_data_sets:
        break

    if INTERRUPTED:
        break

print(f"Info: Parsed {NUMBER_OF_DATA_SETS} events")

# Store data:

print("Info: Storing the data")
with open(output_file_name, "wb") as file_handle:
    pickle.dump(data_sets, file_handle)
print("Info: Done")

#input("Press [enter] to EXIT")
sys.exit(0)
