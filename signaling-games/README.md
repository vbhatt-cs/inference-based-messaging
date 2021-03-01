# Decentralized Learning in Signaling Games

Algorithms to learn communication in one-step cooperative signaling games.

## Installation

- `pip install -r requirements.txt` for regular installation.
- `conda env create -f requirements.yaml` for creating a conda environment.

## Running

- _run()_ function in _train.py_ can be used to train the agents. The function expects a 
  dictionary with all the parameters listed in _default_config.json_ (check 
  `python train.py -h` for documentation about the parameters). 

## Files / Folders

#### Files for running experiments

- _train.py_ - Code for training the agents
- _paper_experiments.py_ - Code to run the experiments described in the paper. Uncomment 
  the required experiment in the _main()_ function. 
- _compare_to_fixed.py_ - Code for comparing an algorithm to its fixed messages and fixed 
  actions variant and plotting/saving.
- _run_single_config.py_ - Code to run a single config from a list of configs. For 
  parallel training on a server, run the required experiment locally using 
  _paper_experiments.py_ till the config list is saved. Then, set the experiment directory
  in _run_single_config.py_ and schedule jobs to run each config.
- _plotting.py_ - Code for plotting. The experiment directory needs to be set inside 
  the required function. The code for saving the figures is commented out.

#### Others

- _algs_ - Implementations of the algorithms
- _utils.py_ - Utility functions
- _plot_utils.py_ - Utility functions for plotting
- _default_config.json_ - Default parameters
- _best_configs_ - Pickled list of the best parameters for each algorithm
- _best_configs_32x32_ - Pickled list of the best parameters for each algorithm in the 
  32x32 setting
  

