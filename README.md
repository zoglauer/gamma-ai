# Enhancing the Data-Analysis Pipeline of Gamma-Ray Telescopes with Machine Learning

## Setup Option 1: Full Installation


### Prerequisites:

* Make sure you have all development tools installed on your computer, such as python3, git, cmake, etc.
* On macOS only the python version downloaded from python.org will work


### Install MEGAlib

Follow [these instructions](http://megalibtoolkit.com/setup.html), to install MEGAlib.


### Get the code

To get the code, just clone the git repository:
```
git clone https://github.com/zoglauer/bids-discovery COSIMachineLearning
```


### Creating the environment

One of the required packages is at the moment only available via pip, not any other python package manager. In addition we need specific versions of some popular packages such as numpy. Therefore, we will have to setup a virtualenv environment to run our specific python version. In the COSIMachineLearning directory, do:

```
python3 -m venv python-env
. python-env/bin/activate
pip3 install -r Requirements.txt
```


### Using it

Remember to activate your python environment whenever you are switching to a new bash shell:
```
. python-env/bin/activate
```

Now you are ready. Switch to the example directory, and take the example to a test drive!


## Setting up Tensorflow with NVIDIA GPU support on Ubuntu 18.04

First, obviously, you need to have a NVIDIA GPU installed in your computer.

### Check versions

Second, determine which version of tensorflow-gpu you want to install. Switch to where you have installed the python environment and do:
```
. python-env/bin/activate
pip install tensorflow-gpu==
```
Pick your version, e.g. 1.13.1. Then lookup the tensorflow webpage, and determine which version of CUDA they recommend:
https://www.tensorflow.org/install/source#linux
Look for the cuDNN and CUDA versions for your chosen tensorflow-gpu version. For 1.13.1 you will need CUDA 10.0 and cuDNN 7.4/


### Install NVIDIA driver

Then install the latest NVIDIA driver.
```
sudo ubuntu-drivers autoinstall
```
In order to be sure that the drivers are in use: reboot.
After the reboot check if your GPU has been identifed and is up and runnign via:
```
nvidia-smi
```

### Install CUDA

Start with verifying that you have at least 4 GB space free in your root partition!

Next head over to the NVIDIA website to install CUDA and cuDNN. 

Install the CUDA version which the website told you (for tensorflow 1.13.1 is was 10.0). Follow these instructions (later on choose deb(local)):
https://developer.nvidia.com/cuda-toolkit-archive

In case a new graphics driver is installed during this process: Reboot!

Set up your path. Add this to your bash configuration file:
```
export PATH=/usr/local/cuda/bin:${PATH} 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
```


Next test if CUDA has been installed correctly
```
cd /usr/local/cuda-<your cuda version>/samples/5_Simulations/nbody
sudo make
./nbody
```


### Install cuDNN

* Install the cuDNN version you determined above. Follow these instructions and pick the right version for your CUDA version (pick the run time library for your linux version): 
https://developer.nvidia.com/rdp/cudnn-archive
After the download, install it:
```
sudo dpkg -i libcudnn7_7.4.2.24-1+cuda10.0_amd64.deb
```

### Lock the packages from unintentional updates

You of course have to adapt it to the packages you have installed, but for the example above, it would be:
```
sudo apt-mark hold cuda libcudnn7 nvidia-driver-410
```
If you want to remove the hold later for an (intentional) update, do:
```
sudo apt-mark unhold cuda libcudnn7 nvidia-driver-410
```


### Ready!

Now you should be ready to go






## Setup Option 2: Docker

This guide will show you how to install and run the tools via a preconfigured docker image.

### Get the source code


To get the code, just clone the git repository:
```
git clone https://github.com/zoglauer/bids-discovery MachineLearning
```

Please remember the full you have cloned the repository to, since you will need it later.


### Install docker and download the docker image

Please follow the guide here to install the latest MEGAlib docker image: [Docker setup](http://megalibtoolkit.com/setup.html#Docker "Docker setup")


### Running the docker image

To run the docker image do (replace /path/to/MachineLearning with the full path of the directory into which you have cloned this repository, and TAG with the tag of the downloaded docker image, e.g. 2.99.11):


#### On Linux:

```
docker run -v /path/to/MachineLearning:/home/mrmegalib/exchange -e DISPLAY=$DISPLAY -e USERID=`id -u ${USER}` -e GROUPID=`id -g ${USER}` -v /tmp/.X11-unix:/tmp/.X11-unix -it zoglauer/megalib:TAG
```

#### On Mac:

```
YOURIP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{ print $2 }')
xhost + ${YOURIP}
docker run --rm -it -v /path/to/MachineLearning:/home/mrmegalib/exchange -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=${YOURIP}:0 -e USERID=`id -u ${USER}` -e GROUPID=`id -g ${USER}` zoglauer/megalib:TAG
```
