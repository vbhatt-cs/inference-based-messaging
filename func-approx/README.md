# Inference-Based Messaging in Function Approximation Settings

Uses [RLlib](https://ray.readthedocs.io/en/latest/rllib.html) for RL algorithms

## Installation

- `pip install -r requirements.txt` for regular installation.
- `conda env create -f requirements.yaml` for creating a conda environment.
- _requirements_cpu.txt_ can be used for CPU-only installation.
- _requirements_analysis.txt_ can be used for analysis-only installation.

## Running
- `python main.py` for training.
- `python eval.py <options>` for visually evaluating the model. Uses 
  [RLlib rollout](https://ray.readthedocs.io/en/latest/rllib-training.html#evaluating-trained-policies) 
  for evaluating. Options are same as those for RLlib rollout.
- `python manual_control.py` for manually controlling the agent. Requires `keyboard` 
  module which is not listed in the requirements.
- Parameters can be set through the corresponding file in `configs/` folder. `main.py` 
  expects a list named `configs` with `configs[0]` being the default config and, 
  optionally, `configs[1:]` being the configs that can be chosen through a command line 
  argument. `main.py` also uses command line arguments and its documentation can be seen 
  by running `python main.py -h`.
- `python analysis.py` for saving the rewards from the RLlib logs. 
  `python analysis.py --analyze` after saving the rewards for calculating and printing the
  mean and the standard error for the experiments.

## Files / Folders
- `envs/` - Code for the environments. Currently, only contains the "treasure hunt" 
  environment.
- `agents/` - Code for the agents. Both single agent and multi-agent policies are modified 
  versions of RLlib's VTraceTFPolicy used in the IMPALA implementation.
- `models/` - Code for the models used by the agents.
- `configs/` - Default configuration for the single agent and multi-agent case.
- `utils/agent_utils.py` - Modifications of RLLib functions used by the agents.
- `utils/model_utils.py` - Helper functions to build neural network layers used in the 
  models.