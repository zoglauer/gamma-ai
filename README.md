# Enhancing the Data-Analysis Pipeline of Gamma-Ray Telescopes with Machine Learning

## Setup Option 1: Full Installation


### Prerequisites:

* Make sure you have all development tools installed on your computer, such as python3, git, cmake, etc.
* Make sure you have installed exactly one instance of python3
* Make sure you have virtualenv installed


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
virtualenv python-env -p python3 --no-site-package
. python-env/bin/activate
pip install -r Requirements.txt
```


### Using it

Remember to activate your python environment whenever you are switching to a new bash shell:
```
. python-env/bin/activate
```

Now you are ready. Switch to the example directory, and take the example to a test drive!


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
