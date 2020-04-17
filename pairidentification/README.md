# Pair identification using machine learning tools

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
cosima -z PairIdentification.source
```
... and wait some time. This just simulated 100,000 triggered events, you might want to have ~10-100 million for the final training run...




## Make the machine learn

The python script PairIdentification.py will perform the machine learning and testing
```
python3 PairIdentification.py -f PairIdentification.inc1.id1.sim.gz -m 10000
```


## To do

* Prevent program from running out of memory 
  * Check if batch size is OK for available memory
  * Stop well before
  * Optimized data storage
* Use the full 3D volume (currently 1-D)
* Swich to vox-net
* Optimize vox-net layout (convolutional layers vs pooling layer, layer parameters)
* Better loss function (cross entropy)
* Our data sets are fully sparse, can we do sparse 3D?

## Run GNN on Savio
Load PyTorch onto Savio.
```
module load pytorch/1.0.0-py36-cuda9.0
```
Running the following command will schedule the training of the GNN.
Change parameters within the shell script to adjust the GPU/CPU allocation, number of events, batch_size, and model architecture.
```
sbatch savio_gnn_test.sh
```
## GNN To do

* Allocate more memory for training
* Use parallel computing to train model faster
  * 100,000+ Events, 100 Batch Size within less than 72 hours
* Save and load previously trained models
* Write visualization scripts
* Make more sophisticated initial connections
* Train different models for different graph sizes

## Save and Restore Model 
To train and save a model, run
```
python3 -u PairIdentificationGNN.py --maxevents 2000 -b 200 --n_iters 10 --save saved_model_state.pt
```
Use .pt file to specify where the model parameters are stored.
To restore the same model and start training again, run
```
python3 -u PairIdentificationGNN.py --maxevents 2000 -b 200 --n_iters 10 --restore saved_model_state.pt
```
Change maxevents, batchsize, and n_iters as needed. 
To save and restore on savio, run 
```
sbatch savio_gnn_test_save.sh
```
and 
```
sbatch savio_gnn_test_restore.sh
```




