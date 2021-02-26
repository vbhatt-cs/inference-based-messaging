# Decentralized Learning in Signaling Games

Algorithms to learn communication in one-step cooperative signaling games.

## Installation

- `pip install -r requirements.txt` for regular installation.
- `conda env create -f requirements.yaml` for creating a conda environment.

## Files / Folders

- _algs_ - Implementations of the algorithms
- _coordination.py_ - Code for running a single instance of the coordination game
- _run_experiments.py_ - Code for comparing an algorithm to its fixed messages and fixed 
  actions variant and plotting/saving
- _utils.py_ - Utility functions
- _plot_utils.py_ - Utility functions for plotting
- _default_config.json_ - Default parameters. Check the help for _coordination.py_ for 
  documentation
- _best_configs_ - Pickled list of the best parameters for each algorithm
- _best_configs_32x32_ - Pickled list of the best parameters for each algorithm in the 
  32x32 setting
