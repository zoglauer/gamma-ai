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
The source file contains a flat input spectrum from 1 to 50 MeV.
Run it via:

```
cosima -z Sim_1MeV_50MeV_flat.source
```
... and wait some time. This just simulated 100,000 triggered events, you might want to have ~10-100 million for the final training run...




## Make the machine learn

The python script EventTypeIdentification.py will perform the machine learning and testing
```
python3 run.py -f 1MeV_50MeV_flat.inc1.id1.sim.gz -a TF:VOXNET -m 10000
```





