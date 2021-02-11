# Energy loss identification using machine learning tools

## Setup

Pleas follow the instruction on the main page on how to setup all the required tools.

If you used virtualenv to install a new python environment, remember to activate it whenever you are switching to a new bash shell:
```
. python-env/bin/activate
```

## Creating a data set

### Simulations

Use the provided simulation source file to create a large data set from which the training and test data set is derived.
The source file contains an isotropic source with flat input spectrum from 0 to 10 MeV.
Run it via:

```
cosima -z Ling.source
```
... and wait some time. This just simulated 100,000 triggered events, you might want to have ~10-100 million for the final training run...



### Preparing the data set

To create a training data set, we have to extract the enhanced Compton scatter information, as well as some evenst quality flags
```
responsecreator -m qf -g ../detectormodel/COSILike.geo.setup -c Ling.revan.cfg -r Ling -f Ling.inc1.id1.sim.gz
```

### Look at the data

To look at the raw data use ROOT:
```
root -x Ling.seq3.quality.root
```
and in the interactive ROOT command prompt, type
```
new TBrowser()
```
On the left pane click on "Ling.seq3.quality.root" --> "Quality;1" and then on any of the histograms below which will be displayed on the right. That's your data.



## Make the machine learn

The python script AlbedoIdentification.py will perform the machine learning and testing
```
python3 run.py -f Ling.seq3.quality.root -o Results -a TMVA:BDT
```

After the machine learning has finished you can look at the results using root:
```
root
```
and then in the ROOT interactive terminal:
```
TMVA::TMVAGui("Results.root");
```
The key plots are 4a, 5, 5a, 5b:

4a shows how good signal and background can be seperated. 5a shows the cut efficiencies and the significance: The higher the peak of the green curve is, the better the approach performs. 5b shows a ROC (receiver-operator-characteristic) curve: The further the curve goes into the top right corner, the better the performance of the approach.





