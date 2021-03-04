# Event type identification using machine learning tools

## Setup

Pleas follow the instruction on the main page on how to setup all the required tools.

If you used virtualenv to install a new python environment, remember to activate it whenever you are switching to a new bash shell:
```
. python-env/bin/activate
```

## Creating a data set 

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


## Notes

Units are keV (kilo-electronvolt), cm (centimeter), and degrees (same for all MEGAlib).



