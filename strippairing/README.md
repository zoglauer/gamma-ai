# Strip pairing using machine learning tools

## Setup

Follow the instreuctions on the main README.md to setup all the required softare tools

## Creating a data set

### Simulations

Use the provided simulation source file to create a large data set from which the training and test data set is derived.
The source file contains an isotropic source with flat input spectrum from 0 to 10 MeV.
Run it via:

```
cosima -z StripPairing.source
```
... and wait some time. This just simulateds 100,000 triggered events, you might want to have ~10-100 million for the final training run...



### Preparing the data set

To create a training data set, we have to extract the strip information as well as the real interaction information:
```
responsecreator -m sf -f StripPairing.inc1.id1.sim.gz -g ../detectormodel/COSILike.geo.setup -c StripPairing.revan.cfg -r StripPairing
```

### Look at the data

To look at the raw data use ROOT:
```
root -x StripPairing.x2.y2.strippairing.root
```
and in the interactive ROOT command prompt, type
```
new TBrowser()
```
On the left pane click on "StripPairing.x2.y2.strippairing.root" --> "StripPairing_2_2" and then on any of the histograms below which will be displayed on the right. That's your data.



## Make the machine learn

The python script StripParing.py will perform the machine learning and testing
```
python3 run.py -f StripPairing.x2.y2.strippairing.root 
```

After this, the result will be printed at the terminal



