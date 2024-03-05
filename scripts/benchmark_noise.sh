#!/bin/bash

# move to script directory for normalized relative paths.
scriptdir="$(dirname "$0")"
cd "$scriptdir"

# original noise setting
python benchmark_noise.py -f ../hyperpars/ppo-asm2o-v0-1.yml -noise 1.5 & 

# lower noise setting
python benchmark_noise.py -f ../hyperpars/ppo-asm2o-v0-1.yml -noise 1.00 & 
python benchmark_noise.py -f ../hyperpars/ppo-asm2o-v0-1.yml -noise 0.75 & 
python benchmark_noise.py -f ../hyperpars/ppo-asm2o-v0-1.yml -noise 0.50 & 
python benchmark_noise.py -f ../hyperpars/ppo-asm2o-v0-1.yml -noise 0.25 & 
python benchmark_noise.py -f ../hyperpars/ppo-asm2o-v0-1.yml -noise 0.10 & 