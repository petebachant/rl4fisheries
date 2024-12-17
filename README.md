# rl4fisheries

Models:

- `asm_env.py`: provides `AsmEnv()`. This encodes our population dynamics model, coupled with an observation process, and a harvest process with a corresponding utility model. These processes can all be modified using the `config` argument. Their defaults are defined in `asm_fns.py`. By default, observations are stock biomass and mean weight.
- `asm_esc.py`: provides `AsmEscEnv()` which inherits from `AsmEnv` and has one difference to it: actions in `AsmEscEnv()` represent escapement levels rather than fishing intensities. 
- `ams_cr_like.py`: provides `AsmCRLike()`. In this environment, mean weight is observed and the action is to set parameters `(x1, x2, y2)` for a biomass-based harvest control rule of the type `CautionaryRule` (specified below).

Strategies evaluated with Bayesian Optimization: 

- `agents.cautionary_rule.CautionaryRule`: piece-wise linear harvest-control rule specified by three parameters `(x1, x2, y2)`. Example plot (TBD).
- `agents.msy.Msy`: constant mortality harvest control rule. Specified by one parameter `mortality`.
- `agents.const_esc.ConstEsc`: constant escapement harvest control rule. Specified by one parameter `escapement`.

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

The Jupyter notebooks found at `rl4fisheries/notebooks/for_results` may be used to recreate the figures found in the paper. 
Notice that the data for the plots is re-generated each time the notebook is run so, e.g., the time-series plots will look different.

To reproduce these figures in your own machine you need to have Jupyter Notebooks installed, however you can navigate to 
```https://github.com/boettiger-lab/rl4fisheries```
and click on `code > codespaces > Create codespace on main` to open the notebooks in a Github codespace.

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

