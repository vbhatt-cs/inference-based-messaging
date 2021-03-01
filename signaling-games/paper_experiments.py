import argparse
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

import algs
from train import run

with open("default_config.json") as f:
    default_config = json.load(f)
    default_config["log_summary"] = True


def create_partial_configs():
    """
    Create a list of partial configs. Payoffs, number of states, messages, and actions
    are added later.
    """
    algorithms = [
        "IQL",
        "IQ",
        "ModelS",
        "ModelR",
        "HystericQ",
        "Lenience",
        "InfoQ",
        "InfoPolicy",
        "CommBias",
    ]
    partial_configs = []
    for alg in algorithms:
        config = default_config.copy()
        config["algorithm"] = alg
        # partial_configs.append(config)
        partial_configs += getattr(algs, alg).generate_configs(config)

    return partial_configs


def run_pc(partial_config, payoffs, n_cpu, exp_dir, exp_num):
    """
    Given a partial config, add the remaining parameters and run the experiments
    """
    # Append all the payoffs to the partial config
    configs = []
    for payoff in payoffs:
        config = partial_config.copy()
        config["payoff"] = payoff
        n_s = int(np.sqrt(len(payoff)))
        config["states"] = n_s
        config["actions"] = n_s
        config["messages"] = n_s
        configs.append(config)

    # Run experiments
    n_cpu = min(len(configs), n_cpu)
    with Pool(n_cpu) as p:
        output = p.map(run, configs)

    # Save the raw output
    with open(exp_dir + "output{}".format(exp_num), "wb") as f:
        pickle.dump(output, f)


def random_payoffs(n_cpu, large=False, tune=True, max_diag=False, exp_dir=None):
    """
    Experiments with random 3x3 or 32x32 payoffs
    Args:
        n_cpu: Number of CPUs to use for parallel running
        large: True if 32x32, false if 3x3
        tune: True if hyperparameters need to be tuned (uses create_partial_configs() to
            generate the list of configs). If False, the function expects the best configs
            to be saved in "best_configs" or "best_configs_32x32" for 3x3 and 32x32
            respectively.
        max_diag: True if the diagonal elements of the payoff matrix should be set to 1
            (not used in the paper)
        exp_dir: (Optional) Set to resume a previous experiment. Expects the directory to
            have at least 3 files: start_i, partial_configs, payoffs. If these files are
            not present, the experiment needs to be re-run
    """
    if exp_dir is not None:
        with open(exp_dir + "start_i", "r") as f:
            start_i = int(f.read())
        with open(exp_dir + "partial_configs", "rb") as f:
            partial_configs = pickle.load(f)
        with open(exp_dir + "payoffs", "rb") as f:
            payoffs = pickle.load(f)
    else:
        exp_dir = "Experiments/PaperExperiments/" + str(int(time.time())) + "/"
        os.makedirs(exp_dir, exist_ok=True)
        if tune:
            partial_configs = create_partial_configs()
            for pc in partial_configs:
                pc["log_interval"] = 1
                pc["episodes"] = 25000 if large else 1000
        else:
            best_config_file = "best_configs_32x32" if large else "best_configs"
            with open(best_config_file, "rb") as f:
                partial_configs = pickle.load(f)
            for pc in partial_configs:
                pc["log_interval"] = 1
                pc["episodes"] = 25000 if large else 1000

        with open(exp_dir + "partial_configs", "wb") as f:
            pickle.dump(partial_configs, f)

        n_payoffs = 100 if tune else 1000
        # n_payoffs = 100
        size = 32 * 32 if large else 3 * 3
        payoffs = np.random.uniform(0, 1, size=(n_payoffs, size))
        if max_diag:
            # Set diagonal elements to 1
            length = 32 if large else 3
            for i in range(length):
                payoffs[:, i * length + i] = 1
        with open(exp_dir + "payoffs", "wb") as f:
            pickle.dump(payoffs, f)

        start_i = 0

    for i in tqdm(range(start_i, len(partial_configs))):
        run_pc(partial_configs[i], payoffs, n_cpu, exp_dir, i)
        with open(exp_dir + "start_i", "w") as f:
            f.write(str(i + 1))


def climbing_game(n_cpu):
    """
    Experiments with the climbing game payoffs. Tuning is not included here since the
    paper uses the same hyperparameters for random 3x3 payoffs and the climbing game. The
    function expects the best configs to be saved in "best_configs".
    Args:
        n_cpu: Number of CPUs to use for parallel running
    """
    exp_dir = "Experiments/PaperExperiments/" + str(int(time.time())) + "/"
    os.makedirs(exp_dir, exist_ok=True)
    with open("best_configs", "rb") as f:
        partial_configs = pickle.load(f)
    with open(exp_dir + "partial_configs", "wb") as f:
        pickle.dump(partial_configs, f)

    for pc in partial_configs:
        pc["log_interval"] = 1

    payoffs = np.array([[11, -30, 0, -30, 7, 6, 0, 0, 5]])

    for i in tqdm(range(len(partial_configs))):
        run_pc(partial_configs[i], payoffs, n_cpu, exp_dir, i)
        with open(exp_dir + "start_i", "w") as f:
            f.write(str(i + 1))


def fixed_baseline(n_cpu, payoff_dir):
    """
    Experiments using Q-Learning with fixed messages or fixed actions. The function
    expects the best configs to be saved in "best_configs".
    Args:
        n_cpu: Number of CPUs to use for parallel running
        payoff_dir: Directory which contains the list of payoff matrices. If not present,
            defaults to the climbing game payoffs.
    """
    exp_dir = "Experiments/PaperExperiments/" + str(int(time.time())) + "/"
    os.makedirs(exp_dir, exist_ok=True)
    with open("best_configs", "rb") as f:
        best_partial_configs = pickle.load(f)

    try:
        with open(payoff_dir + "payoffs", "rb") as f:
            payoffs = pickle.load(f)
    except FileNotFoundError:
        payoffs = np.array([[11, -30, 0, -30, 7, 6, 0, 0, 5]])

    partial_config = None

    for pc in best_partial_configs:
        if pc["algorithm"] == "IQL":
            pc["log_interval"] = 1
            partial_config = pc

    partial_configs = []
    for mode in [1, 2]:
        pc = partial_config.copy()
        pc["mode"] = mode
        pc["eps0"] = 1.0
        pc["eps0_decay"] = 1.0 / (pc["episodes"] * 0.8)
        partial_configs.append(pc)

    with open(exp_dir + "partial_configs", "wb") as f:
        pickle.dump(partial_configs, f)

    with open(exp_dir + "payoffs", "wb") as f:
        pickle.dump(payoffs, f)

    for i in tqdm(range(len(partial_configs))):
        run_pc(partial_configs[i], payoffs, n_cpu, exp_dir, i)
        with open(exp_dir + "start_i", "w") as f:
            f.write(str(i + 1))


def main():
    parser = argparse.ArgumentParser(description="Experiments")
    parser.add_argument(
        "--n-cpu", type=int, default=5, metavar="N", help="Number of CPUs available"
    )
    args = parser.parse_args()
    # random_payoffs(args.n_cpu, tune=False)
    # random_payoffs(args.n_cpu, large=True, tune=False)
    # climbing_game(args.n_cpu)
    # fixed_baseline(args.n_cpu, "Experiments/PaperExperiments/{}/".format("1591036032"))
    # fixed_baseline(args.n_cpu, "Experiments/PaperExperiments/{}/".format("1591032779"))
    # fixed_baseline(args.n_cpu, "Experiments/PaperExperiments/{}/".format("32x32"))


if __name__ == "__main__":
    main()
