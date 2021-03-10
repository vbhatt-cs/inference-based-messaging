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

- `train.py` - Code for training the agents
- `paper_experiments.py` - Code to run the experiments described in the paper. Uncomment 
  the required experiment in the `main()` function. 
- `compare_to_fixed.py` - Code for comparing an algorithm to its fixed messages and fixed 
  actions variant and plotting/saving.
- `run_single_config.py` - Code to run a single config from a list of configs. For 
  parallel training on a server, run the required experiment locally using 
  `paper_experiments.py` till the config list is saved. Then, set the experiment directory
  in `run_single_config.py` and schedule jobs to run each config.
- `plotting.py` - Code for plotting. The experiment directory needs to be set inside 
  the required function. The code for saving the figures is commented out.

#### Others

- `algs/` - Implementations of the algorithms
- `utils.py` - Utility functions
- `plot_utils.py` - Utility functions for plotting
- `default_config.json` - Default parameters
- `best_configs` - Pickled list of the best parameters for each algorithm
- `best_configs_32x32` - Pickled list of the best parameters for each algorithm in the 
  32x32 setting
  

