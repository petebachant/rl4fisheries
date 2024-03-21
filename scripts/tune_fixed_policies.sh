#!/bin/bash

# move to script directory for normalized relative paths.
scriptdir="$(dirname "$0")"
cd "$scriptdir"

# hf
python login.py

# gp
python fixed_policy_opt.py -p msy -v True -o gp -nc 100 &
python fixed_policy_opt.py -p esc -v True -o gp -nc 100 &
python fixed_policy_opt.py -p cr -v True -o gp -nc 100 &

# gbrt
python fixed_policy_opt.py -p msy -v True -o gbrt -nc 100 &
python fixed_policy_opt.py -p esc -v True -o gbrt -nc 100 &
python fixed_policy_opt.py -p cr -v True -o gbrt -nc 100 &