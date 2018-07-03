# BIDS discovery project 

## General setup


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
