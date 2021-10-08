# Event type identification using machine learning tools

## Setup

Pleas follow the instruction on the main page on how to setup all the required tools.

If you used virtualenv to install a new python environment, remember to activate it whenever you are switching to a new bash shell:
```
. python-env/bin/activate
```

## Creating a data set 

### Update as of 9/29/21

To create a data set from which the training and testing data sets can pull, use event_extractor.py, which in turn relies on event_data.py.
The two together allow the data organization and parsing to be completely separate from the main file, EnergyLossEstimate.py. However,
the prior structure allowing an easy switch between the models to be tested has been retained; currently, this switch is done via a
command line argument when running EnergyLossEstimate.py. Both event_data.py and event_extractor.py have docstrings explaining their
contents in more depth. - Auden Young

### Simulations

Use the provided simulation source file to create a large data set from which the training and test data set is derived.
The source file contains a flat input spectrum from 2 to 1000 MeV.
Run it via:

```
cosima -z Sim_2MeV_1000MeV_flat.source
```
... and wait some time. This just simulated 100,000 triggered events, you might want to have ~10-100 million for the final training run...


## Make Plots

python3 run.py -f SIM_FILE_PATH -p True

Optionally add "-m N" where N = number of events. Default N=100,000.

## Make the machine learn

The python script run.py will perform the machine learning and testing
```
python3 run.py -f 2MeV_1000MeV_flat.inc1.id1.sim.gz -a TF:VOXNET -m 10000
```

The machine learning model can be chosen using the algorithm (-a) option when running. Current algorithm options are a voxnet, voxnet with batch normalization, and voxnet with layer normalization. Each model is based and trained using the hits data, and predicts the gamma energy. We currently handle two different event types.


## Training on Savio
To train the model on the Savio cluster, the provided .sh file can be used. Submit a job for training on Savio via:
```
sbatch SavioSubmitScript_SingleGPU.sh
```
Before running this command, verify that the -a option of EnergyLossEstimate.py indicates the algorithm to be used for training.


## Notes

Units are keV (kilo-electronvolt), cm (centimeter), and degrees (same for all MEGAlib).

