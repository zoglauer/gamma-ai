import argparse
import os
import pickle
import sys


def parseTrainingData():
    """Function to parse simulation data file into event_list."""

    # parse training file
    training_file = 'EnergyLoss.10k.v1.data' #toggle between 10k and 100k as desired
    parser = argparse.ArgumentParser(
        description='Perform training and/or testing of the event clustering machine learning tools.')
    parser.add_argument('-f', '--filename', default=training_file,
                        help='File name used for training/testing')
    parser.add_argument('-s', '--savefileto', default='shower_output/shower_events.pkl',
                        help='save file name for event data with shower profile estimates.')

    args = parser.parse_args()

    file_name = args.filename

    # no path to specified file
    if file_name and not os.path.exists(file_name):

        print(f"Error: The training data file does not exist: {file_name}")
        sys.exit(0)

    # file found
    elif file_name:

        print(f"CMD: Using file {file_name}")

        # get event list
        with open(file_name, "rb") as file_handle:
            event_list = pickle.load(file_handle)

        return event_list

    print("No file found.")
    sys.exit(0)

def get_num_files(directory, name):
    files = os.listdir(directory)
    num_files = len([file for file in files if file == name])
    return num_files
