import argparse
import pickle

import numpy as np

from train import run


def run_exp(n, exp_dir=None):
    with open(exp_dir + "partial_configs", "rb") as f:
        partial_configs = pickle.load(f)
    with open(exp_dir + "payoffs", "rb") as f:
        payoffs = pickle.load(f)

    n_payoff = n % payoffs.shape[0]
    n_config = n // payoffs.shape[0]
    print(n_config, n_payoff)

    config = partial_configs[n_config].copy()
    config["payoff"] = payoffs[n_payoff]
    n_s = int(np.sqrt(payoffs.shape[1]))
    config["states"] = n_s
    config["actions"] = n_s
    config["messages"] = n_s
    output = run(config)

    # Save the raw output
    with open(exp_dir + "output{}_{}".format(n_config, n_payoff), "wb") as f:
        pickle.dump(output, f)


def main():
    parser = argparse.ArgumentParser(description="Experiments")
    parser.add_argument("--n", type=int, metavar="N", help="Number of the experiment")
    args = parser.parse_args()
    run_exp(args.n, "Experiments/PaperExperiments/32x32/")


if __name__ == "__main__":
    main()
