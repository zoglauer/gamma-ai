#!/bin/bash

PENV=python-env

if [[ ${HOSTNAME} == thebe ]]; then
  export TMPDIR=/volumes/selene/tmp
fi

if [ -d ${PENV} ]; then
  rm -r ./${PENV}
fi
python3 -m venv ${PENV}
. ${PENV}/bin/activate
pip3 install -r Requirements.txt

if [[ ${HOSTNAME} == thebe ]] || [[ ${HOSTNAME} == despina ]]; then
  pip install tensorflow-gpu==2.3
fi


