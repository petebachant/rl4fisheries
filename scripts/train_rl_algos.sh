#!/bin/bash

# move to script directory for normalized relative paths.
scriptdir="$(dirname "$0")"
cd "$scriptdir"

# just for good measure, install
pip install -e ..

# hf
python hf_login.py

# train
python train.py -f ../hyperpars/ppo-asm.yml &
python train.py -f ../hyperpars/tqc-asm.yml &
python train.py -f ../hyperpars/rppo-asm.yml &