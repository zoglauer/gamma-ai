import pickle, argparse
import os, sys


def parseTrainingData():
    """Function to parse simulation data file into event_list."""

    # parse training file
    parser = argparse.ArgumentParser(
        description='Perform training and/or testing of the event clustering machine learning tools.')
    parser.add_argument('-f', '--filename', default='EnergyEstimate.p1.sim.gz',
                        help='File name used for training/testing')
    parser.add_argument('-s', '--savefileto', default='shower_output/shower_events.pkl',
                        help='save file name for event data with shower profile estimates.')

    args = parser.parse_args()

    # check if file exists
    if args.filename != "":
        file_name = args.filename
    if not os.path.exists(file_name):
        print(f"Error: The training data file does not exist: {file_name}")
        sys.exit(0)
    print(f"CMD: Using file {file_name}")

    # get event list
    with open(file_name, "rb") as file_handle:
        event_list = pickle.load(file_handle)

    return event_list