# BIDS discovery project 

## General setup


### Prerequisites:

* Make sure you have all development tools installed on your computer, such as python, git, etc.
* Make sure you have MEGAlib installed (test it by running a small simulation by just launching cosima in the command prompt)
* Since we do not want to overburden our python environment, we use virtualenv to create a local python setup. For that make sure virtualenv is installed. If it is not installed installed it via your prefered tool (e.g. for anaconda: "conda install virtualenv", or for pip: "pip install virtualenv")

That's all.


### Get the code


To get the code, just clone the git repository:
```
git clone https://guthub.com/zoglauer/bids-discovery Discovery
```

### Creating the environment

```
virtualenv python-env
. python-env/bin/activate
pip install rootpy
```

Remember to activate your python environment whenever you are switching to a new bash shell:
```
. python-env/bin/activate
```

Now you are ready. Switch to the example directory, and take the example to a test drive!