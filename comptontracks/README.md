# Compton identification using machine learning tools

## Setup

Pleas follow the instruction on the main page on how to setup all the required tools.

If you used virtualenv to install a new python environment, remember to activate it whenever you are switching to a new bash shell:
```
. python-env/bin/activate
```

## Creating a data set

### Simulations

Use the provided simulation source file to create a large data set from which the training and test data set is derived.
The source file contains a flat input spectrum from 0.5 to 10 MeV.
Run it via:

```
cosima -z ComptonTrackIdentification.source
```
... and wait some time. This just simulated 100,000 triggered events, you might want to have ~10-100 million for the final training run...




## Make the machine learn

The python script ComptonTrackIdentification.py will perform the machine learning and testing
```
python3 ComptonTrackIdentification.py -f ComptonTrackIdentification.inc1.id1.sim.gz -m 10000
```


## Remaining To Do List

### First and second position
* + Switch back to full 3D
* + Reactivate vox-net
* + Switch to cross-entropy loss function
* - Optimize vox-net
* + Use all track lengths (currently fixed to 5 layers)

### Find all hits belonging to the track and thus the full energy
* Needs different network layout: Need to classify each hit if it belongs to the Compton track

### Overall
* +/- Can we switch to sparse data storage?
