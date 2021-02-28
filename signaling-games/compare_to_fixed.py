import argparse
import bz2
import json
import os
import pickle
import time
from multiprocessing import Pool

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from train import run
from plot_utils import plot_confusion_matrix, plot_venn

with open("default_config.json") as f:
    default_config = json.load(f)

color_sequence = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
]
params = {
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
mpl.rcParams.update(params)


def main(n_cpu, save=True, exp_dir=None):
    configs = []
    for mode in range(3):
        config = default_config.copy()
        config["mode"] = mode
        configs.append(config)

    # Run experiments
    n_cpu = min(len(configs), n_cpu)
    with Pool(n_cpu) as p:
        output = p.map(run, configs)
    # output = run(default_config)

    if save:
        exp_dir = (
            "Experiments/" + str(int(time.time())) + "/" if exp_dir is None else exp_dir
        )
        os.makedirs(exp_dir, exist_ok=True)

    # Plot reward graph and save it
    fig1, ax1 = plt.subplots()
    legends = ["Communication", "Fixed messages", "Fixed actions"]
    for i, (rs, cps, ns) in enumerate(output):
        ax1.set_ylim(-0.2, 1.05)
        ax1.set_ylabel("Obtained reward / max reward for the state")
        ax1.set_xlabel("Episodes")
        ys = rs.mean(axis=1)
        rs_sem = stats.sem(rs, axis=1)
        xs = np.arange(len(ys)) * default_config["log_interval"]
        ax1.plot(xs, ys, color=color_sequence[i], label=legends[i])
        plt.fill_between(xs, ys - rs_sem, ys + rs_sem, alpha=0.5, color=color_sequence[i])
        ax1.legend()
    if save:
        fig1.savefig(exp_dir + "reward_plot")

    # Plot the convergence points and save it
    fig2, ax2 = plot_confusion_matrix(output[0][1].astype(np.int))
    if save:
        fig2.savefig(exp_dir + "cp_plot")

    # Plot the venn diagram for messages and save it
    fig3, ax3 = plot_venn(output[0][2])
    if save:
        fig3.savefig(exp_dir + "message_plot")

    # Save the configs and the raw output
    if save:
        with open(exp_dir + "configs", "wb") as f:
            pickle.dump(configs, f)
        with bz2.BZ2File(exp_dir + "output.pbz2", "wb") as f:
            pickle.dump(output, f)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments")
    parser.add_argument(
        "--n-cpu", type=int, default=5, metavar="N", help="Number of CPUs available"
    )
    args = parser.parse_args()
    main(args.n_cpu, save=default_config["save"])
