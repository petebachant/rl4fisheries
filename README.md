# rl4fisheries

RL and Bayesian optimization methodologies for harvest control rule optimization in fisheries.
Includes:
- A gymnasium environment for a Walleye population dynamics model
- Policy functions for different commonly-tested policies (including those in the paper)
- Scripts to optimize RL policies and non-RL policies
- Notebooks to reproduce paper figures
- Templates to train new RL policies on our Walleye environment

## Installation

To install this source code, you need to have git, Python and pip installed.
To quickly check whether these are installed you can open the terminal and run the following commands:
```bash
git version
pip --version
python -V
```
If the commands are not recognized by the terminal, refer to
[here](https://github.com/git-guides/install-git)
for git installation instructions,
[here](https://realpython.com/installing-python/) 
for Python installation instructions and/or
[here](https://pip.pypa.io/en/stable/installation/)
for pip installation instructions.

To install this source code, run
```bash
git clone https://github.com/boettiger-lab/rl4fisheries.git
cd rl4fisheries
pip install -e .
```

## Optimized policies

The optimized policies presented in the paper---both RL policies and non-RL policies such as the precautionary policy---are saved in a public hugging-face 
[repository](https://huggingface.co/boettiger-lab/rl4eco/tree/main/sb3/rl4fisheries/results).
RL policies are saved as zip files named ```PPO-AsmEnv-(...)-UMx-(...).zip``` since the RL algorithm PPO was used to optimize them.
Here *UM* stands for *utility model* and `x=1, 2, or 3` designates which utility model the policy was optimized for.
Precautionary policies are named `cr-UMx.pkl` (CR stands for "cautionary rule", an acronym we used during the research phase of this collaboration).
Similarly, constant escapement policies are saved as `esc-UMx.pkl` and FMSY policies are saved as `msy-UMx.pkl`.

## Reproducing paper figures

The Jupyter notebooks found at `rl4fisheries/notebooks/for_results` may be used to re-generate the csv data used in our figures.
Notice that the data for the plots is re-generated each time the notebook is run so, e.g., the time-series plots will look different due to stochasticity.

The specific data used to generate the figures in our paper, along with the Jupyter notebooks to construct the figures and make them look pretty, can be found in this huggingface repo:

```https://huggingface.co/datasets/felimomo/rl4fisheries-public-data```

## Optimizing RL policies

To optimize an RL policy from scratch, use the command
```bash
python scripts/train.py -f path/to/config/file.yml
```
You can use the following template config file:
```bash
python scripts/train.py -f hyperpars/RL-template.yml
```

The config files we used for the policies in our paper are found at `hyperpars/for_results/`.
For example 
[this](https://github.com/boettiger-lab/rl4fisheries/blob/main/hyperpars/for_results/ppo_biomass_UM1.yml) 
config file was used to train 1-Obs. RL in Scenario 1 (utility = total harvest).
The trained model is automatically pushed to hugging-face if a hugging-face token is provided. 

## Source code structure

```
rl4fisheries
|
|-- hyperpars
|   |
|   |-- configuration yaml files
|
|-- notebooks
|   |
|   |-- Jupyter notebooks
|
|-- src/rl4fisheries
|   |
|   |-- agents
|   |   |
|   |   |-- interfaces for policies such as Precautionary Policy, FMSY, Constant Escapement
|   |
|   |-- envs
|   |   |
|   |   |-- Gymnasium environments used in our study.
|   |       (specifically, asm_env.py is used for our paper).
|   |
|   |-- utils
|       |
|       |-- ray.py: RL training within Ray framework (not used in paper)
|       |
|       |-- sb3.py: RL training within Stable Baselines framework (used in paper)
|       |
|       |-- simulation.py: helper functions to simulate the system dynamics using a policy
|    
|-- tests
|   |
|   |-- continuous integration tests to ensure code quality in pull requests
|
|-- noxfile.py: file used to run continuous integration tests
|
|-- pyproject.toml: file used in the installation of this source code
|
|-- README.md 
```
