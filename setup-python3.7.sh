#!/bin/bash

#Setup: 
# sudo add-apt-repository ppa:deadsnakes/ppa
# sudo apt update 
# sudo apt-get install python3.7  python3.7-venv python3.7-dev

if [[ ! -d python-env ]]; then
  python3.7 -m venv python-env
  . python-env/bin/activate
  pip3 install -r Requirements.txt
else
  . python-env/bin/activate
fi


