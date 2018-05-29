# Event clustering using machine learning tools

## Setup

First install MEGAlib (see https://github.com/zoglauer/meglib). Make sure you are on the experimental branch, which should give you at least ROOT 6.10 -- you can test it with 
```
root-config --version
```
Then set up the python environment. My suggestion would be to use virtual-env to avoid overburdening your python environment:
```
virtualenv python-env
. python-env/bin/activate
pip install rootpy
```

Remember to activate your python environment whenever you are switching to a new bash shell:
```
. python-env/bin/activate
```

## Creating a data set

### Simulations

Use the provided simulation source file to create a large data set from which the training and test data set is derived.
The source file contains an isotropic source with flat input spectrum from 0 to 10 MeV.
Run it via:

```
cosima -z EC.source
```
... and wait some time. This just simulated 100,000 triggered events, you might want to have ~10-100 million for the final training run...



### Preparing the data set

To create a training data set, we have to extract the enhanced Compton scatter information, as well as some evenst quality flags
```
responsecreator -g ../detectormodel/COSILike.geo.setup -f EC.inc1.id1.sim -r EC -m gf -c EC.revan.cfg -z -o maxhits=10:maxgroups=3
```
This should create 10 ROOT files: EC.maxhits1.eventclusterizer.root, ..., EC.maxhits10.eventclusterizer.root 

### Look at the data

To look at the raw data use ROOT:
```
root -x EC.maxhits3.eventclusterizer.root
```
and in the interactive ROOT command prompt, type
```
new TBrowser()
```



## Make the machine learn

The python script AlbedoIdentification.py will perform the machine learning and testing
```
python EC.py
```

After it is done you can look at the results using root:
```
root
```
and then in the ROOT interactive terminal:
```
TMVA::TMVAGui("Results.root");
```
The key plots are 4a, 5a, 5b



