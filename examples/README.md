# Running the examples:

First install MEGAlib (see https://github.com/zoglauer/meglib). Make sure you are on the experimental branch, which should give you at least ROOT 6.08 -- you can test it with 
```
root-config --version
```
Then set up the python environment. My suggestion would be to use virtual-env to avoid overburdening your python environment:
```
virtualenv python-env
. python-env/bin/activate
pip install rootpy
```

Finally, run the examples via:
```
python TMVA.py
```

Now you can look at the results using root:
```
root
```
and then in the ROOT interactive terminal:
```
TMVA::TMVAGui("Results.root");
```
The key plots are 4a, 5a, 5b


Finally, remember to activate your python environment whenever you are switching to a new bash shell:
```
. python-env/bin/activate
```

