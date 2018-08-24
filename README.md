# Enhancing the data analysis pipeline of Gamma-ray telescopes with machine learning

## Setup option 1: Docker

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


## Setup option 2: Full installation


### Prerequisites:

* Make sure you have all development tools installed on your computer, such as python, git, etc. On the Mac, make sure you have a somewhat clean environment -- the success of a full installation is reduced drastically if you have installed a few additional python version, e.g. via anaconda, miniconda, etc.
* Make sure you have MEGAlib installed (test it by running a small simulation by just launching cosima in the command prompt)
* Since we do not want to overburden our python environment, we use virtualenv to create a local python setup. For that make sure virtualenv is installed. If it is not installed installed it via your prefered tool (e.g. for anaconda: "conda install virtualenv", or for pip: "pip install virtualenv")

That's all.


### Get the code


To get the code, just clone the git repository:
```
git clone https://github.com/zoglauer/bids-discovery Discovery
```

### Creating the environment

The first thing to do is to check which python version root has been compiled with - you have to use the same version for this task. Execute:

```
root-config --config
```

If you find something like python2 the use Python 2, if you find something like python3, then use Python 3. If you find no mention of Python, then please reinstall ROOT with python support enabled.

#### Python 2

```
virtualenv python-env -p python2 --no-site-package
. python-env/bin/activate
pip install rootpy
```

#### Python 3

```
virtualenv python-env -p python3 --no-site-package
. python-env/bin/activate
pip install rootpy
```

#### If it did not work

If you get an error message like:
```
ROOT cannot be imported. Is ROOT installed with PyROOT enabled?
```
then first look at 
```
root-config --features
```
In the output, python and/or python3 should appear. If both or only python3 appears, try the following: 
```
pip3 install rootpy
```

### Using it

Remember to activate your python environment whenever you are switching to a new bash shell:
```
. python-env/bin/activate
```

Now you are ready. Switch to the example directory, and take the example to a test drive!
