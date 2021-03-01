import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import stats
from tqdm import tqdm

from plot_utils import plot_confusion_matrix, plot_venn

color_sequence = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#999999",
]
color_sequence_swapped = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#999999",
    "#a65628",
    "#f781bf",
]
params = {
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
}
mpl.rcParams.update(params)


def init_fig():
    fig, ax = plt.subplots(1, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def random_payoffs(exp_dir):
    """
    Plotting for experiments with random 3x3 and 32x32 payoffs if the results were
    stored by paper_experiments.py
    """
    # exp_dir = "Experiments/PaperExperiments/{}/".format("1591032779")
    # exp_dir = "Experiments/PaperExperiments/{}/".format("1596153082")
    baseline_dir = "Experiments/PaperExperiments/{}/".format("1598034745")

    with open(exp_dir + "partial_configs", "rb") as f:
        partial_configs = pickle.load(f)
    with open(exp_dir + "payoffs", "rb") as f:
        payoffs = pickle.load(f)
    with open(baseline_dir + "partial_configs", "rb") as f:
        baseline_partial_configs = pickle.load(f)

    n_payoffs = 1000
    percent_opts = []
    fig1, ax1 = init_fig()
    fig12, ax12 = init_fig()
    legends = [
        "IQL",
        "IQ",
        "ModelS",
        "ModelR",
        "Hysteretic-Q",
        "Lenience",
        "Info-Q",
        "Info-Policy",
        "Comm-Bias",
    ]
    baseline_legends = [
        "IQL",
        "InfoQ",
        "InfoPolicy",
        "Fixed messages",
        "Fixed actions",
    ]
    legends_swapped = [
        "IQL",
        "IQ",
        "ModelS",
        "ModelR",
        "Hysteretic-Q",
        "Lenience",
        "Comm-Bias",
        "Info-Q",
        "Info-Policy",
    ]
    baseline_i = 0
    for i in range(len(partial_configs)):
        with open(exp_dir + "output{}".format(i), "rb") as f:
            op = pickle.load(f)
            rs_mean, rs_sem = split_mean(0, op, 10)
            xs = np.arange(len(rs_mean)) * 10
            ax1.plot(xs, rs_mean, color=color_sequence[i], label=legends[i])
            ax1.fill_between(
                xs, rs_mean - rs_sem, rs_mean + rs_sem, alpha=0.5, color=color_sequence[i]
            )
            if partial_configs[i]["algorithm"] in ["IQL", "InfoQ", "InfoPolicy"]:
                ax12.plot(
                    xs,
                    rs_mean,
                    color=color_sequence[baseline_i],
                    label=baseline_legends[baseline_i],
                )
                ax12.fill_between(
                    xs,
                    rs_mean - rs_sem,
                    rs_mean + rs_sem,
                    alpha=0.5,
                    color=color_sequence[baseline_i],
                )
                baseline_i += 1

            percent_opt = [op[j][4] * 100 for j in range(len(op))]
            percent_opts.append(percent_opt)

    for i in range(len(baseline_partial_configs)):
        with open(baseline_dir + "output{}".format(i), "rb") as f:
            op = pickle.load(f)
            rs_mean, rs_sem = split_mean(0, op, 10)
            xs = np.arange(len(rs_mean)) * 10
            ax12.plot(
                xs,
                rs_mean,
                color=color_sequence[baseline_i],
                label=baseline_legends[baseline_i],
            )
            ax12.fill_between(
                xs,
                rs_mean - rs_sem,
                rs_mean + rs_sem,
                alpha=0.5,
                color=color_sequence[baseline_i],
            )
            baseline_i += 1

    xs = [0, 1000, 800, 800, 280, 800, 200, -30, 1000]
    ys = [0, 0.885, 0.93, 1.9, 1.945, 0.82, 1, 0.95, 1.9]
    for i in range(len(legends)):
        color = color_sequence[i] if i != 5 else "#000000"
        ax1.text(xs[i], ys[i], legends[i], color=color)

    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Normalized reward")

    xs = [0, 1000, 800, 800, 280]
    ys = [0, 0.885, 0.93, 1.9, 1.945]
    for i in range(len(baseline_legends)):
        color = color_sequence[i] if i != 5 else "#000000"
        ax12.text(xs[i], ys[i], baseline_legends[i], color=color)

    ax12.set_xlabel("Episodes")
    ax12.set_ylabel("Normalized reward")

    # fig1.savefig(exp_dir + "norm_reward.pdf", format="pdf")

    # Swapping to keep comm-bias before info-Q
    po = percent_opts.pop(-1)
    percent_opts.insert(6, po)

    fig2, ax2 = init_fig()
    ax2.tick_params(axis="x", bottom=False, labelbottom=False)
    ax2.set_ylabel("% of runs converged to optimal policy")
    bplot = ax2.boxplot(percent_opts, notch=True, patch_artist=True)
    for patch, color in zip(bplot["boxes"], color_sequence_swapped):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    for i, alg in enumerate(legends_swapped):
        if i >= 7:
            ax2.text(i + 0.9, 105, alg, rotation=90, fontweight="bold")
        else:
            ax2.text(i + 0.9, 105, alg, rotation=90)

    # fig2.savefig(exp_dir + "boxplot.pdf", format="pdf", bbox_inches="tight")


def climbing_game():
    """
    Plotting for experiments with the climbing game payoffs
    """
    exp_dir = "Experiments/PaperExperiments/{}/".format("1591036032")
    baseline_dir = "Experiments/PaperExperiments/{}/".format("1598035516")

    with open(exp_dir + "partial_configs", "rb") as f:
        partial_configs = pickle.load(f)
    with open(baseline_dir + "partial_configs", "rb") as f:
        baseline_partial_configs = pickle.load(f)

    fig11, ax11 = init_fig()
    fig112, ax112 = init_fig()
    fig12, ax12 = init_fig()
    fig122, ax122 = init_fig()
    cps = []
    ns = []
    legends = [
        "IQL",
        "IQ",
        "ModelS",
        "ModelR",
        "Hysteretic-Q",
        "Lenience",
        "Info-Q",
        "Info-Policy",
        "Comm-Bias",
    ]
    baseline_legends = [
        "IQL",
        "Info-Q",
        "Info-Policy",
        "Fixed messages",
        "Fixed actions",
    ]
    baseline_i = 0
    for i in range(len(partial_configs)):
        with open(exp_dir + "output{}".format(i), "rb") as f:
            op = pickle.load(f)
            rs_mean, rs_sem = split_mean(0, op, 10)
            opts_mean, opts_sem = split_mean(2, op, 10)
            cps.append(op[0][-2])
            ns.append(op[0][-1])

            xs = np.arange(len(rs_mean)) * 10
            ax11.plot(xs, rs_mean, color=color_sequence[i], label=legends[i])
            ax11.fill_between(
                xs, rs_mean - rs_sem, rs_mean + rs_sem, alpha=0.5, color=color_sequence[i]
            )
            ax12.plot(xs, opts_mean * 100, color=color_sequence[i], label=legends[i])
            ax12.fill_between(
                xs,
                (opts_mean - opts_sem) * 100,
                (opts_mean + opts_sem) * 100,
                alpha=0.5,
                color=color_sequence[i],
            )

            if partial_configs[i]["algorithm"] in ["IQL", "InfoQ", "InfoPolicy"]:
                ax112.plot(
                    xs,
                    rs_mean,
                    color=color_sequence[baseline_i],
                    label=baseline_legends[baseline_i],
                )
                ax112.fill_between(
                    xs,
                    rs_mean - rs_sem,
                    rs_mean + rs_sem,
                    alpha=0.5,
                    color=color_sequence[baseline_i],
                )
                ax122.plot(
                    xs,
                    opts_mean * 100,
                    color=color_sequence[baseline_i],
                    label=baseline_legends[baseline_i],
                )
                ax122.fill_between(
                    xs,
                    (opts_mean - opts_sem) * 100,
                    (opts_mean + opts_sem) * 100,
                    alpha=0.5,
                    color=color_sequence[i],
                )
                baseline_i += 1

    for i in range(len(baseline_partial_configs)):
        with open(baseline_dir + "output{}".format(i), "rb") as f:
            op = pickle.load(f)
            rs_mean, rs_sem = split_mean(0, op, 10)
            opts_mean, opts_sem = split_mean(2, op, 10)
            xs = np.arange(len(rs_mean)) * 10
            ax112.plot(
                xs,
                rs_mean,
                color=color_sequence[baseline_i],
                label=baseline_legends[baseline_i],
            )
            ax112.fill_between(
                xs,
                rs_mean - rs_sem,
                rs_mean + rs_sem,
                alpha=0.5,
                color=color_sequence[baseline_i],
            )
            ax122.plot(
                xs,
                opts_mean * 100,
                color=color_sequence[baseline_i],
                label=baseline_legends[baseline_i],
            )
            ax122.fill_between(
                xs,
                (opts_mean - opts_sem) * 100,
                (opts_mean + opts_sem) * 100,
                alpha=0.5,
                color=color_sequence[i],
            )
            baseline_i += 1

    xs = [0, 1000, 800, 800, 280, 800, 200, -30, 0]
    ys = [0, 0.885, 0.93, 1.9, 1.945, 0.82, 1, 0.95, 1.9]
    for i in range(len(legends)):
        color = color_sequence[i] if i != 5 else "#000000"
        ax11.text(xs[i], ys[i], legends[i], color=color)

    xs = [0, 1000, 800, 800, 280, 800, 200, -30, 0]
    ys = np.array([0, 0.885, 0.93, 1.9, 1.945, 0.82, 1, 0.95, 1.9]) * 100
    for i in range(len(legends)):
        color = color_sequence[i] if i != 5 else "#000000"
        ax12.text(xs[i], ys[i], legends[i], color=color)

    ax11.set_xlabel("Episodes")
    ax11.set_ylabel("Normalized reward")
    ax12.set_xlabel("Episodes")
    ax12.set_ylabel("% of optimal actions")

    xs = [0, 1000, 800, 800, 280]
    ys = [0, 0.885, 0.93, 1.9, 1.945]
    for i in range(len(baseline_legends)):
        color = color_sequence[i] if i != 5 else "#000000"
        ax112.text(xs[i], ys[i], baseline_legends[i], color=color)

    ax112.set_xlabel("Episodes")
    ax112.set_ylabel("Normalized reward")

    xs = [0, 1000, 800, 800, 280]
    ys = [0, 0.885, 0.93, 1.9, 1.945]
    for i in range(len(baseline_legends)):
        color = color_sequence[i] if i != 5 else "#000000"
        ax122.text(xs[i], ys[i], baseline_legends[i], color=color)

    ax122.set_xlabel("Episodes")
    ax122.set_ylabel("% of optimal actions")

    # fig11.savefig(exp_dir + "norm_reward.pdf", format="pdf")
    # fig12.savefig(exp_dir + "opts.pdf", format="pdf")

    fig21, ax21 = plot_confusion_matrix(cps[0].astype(np.int))
    fig22, ax22 = plot_confusion_matrix(cps[-2].astype(np.int))

    # fig21.savefig(exp_dir + "cp_iql.pdf", format="pdf", bbox_inches="tight")
    # fig22.savefig(exp_dir + "cp_infoq.pdf", format="pdf", bbox_inches="tight")

    fig31, ax31 = plot_venn(ns[0])
    fig32, ax32 = plot_venn(ns[-2])

    # fig31.savefig(exp_dir + "venn_iql.pdf", format="pdf", bbox_inches="tight")
    # fig32.savefig(exp_dir + "venn_infoq.pdf", format="pdf", bbox_inches="tight")


def cc_random_payoffs():
    """
    Plotting for experiments with random 3x3 and 32x32 experiments if the results were
    stored by run_single_config.py
    """
    exp_dir = "Experiments/PaperExperiments/{}/".format("32x32")
    baseline_dir = "Experiments/PaperExperiments/{}/".format("1598034745")

    with open(exp_dir + "partial_configs", "rb") as f:
        partial_configs = pickle.load(f)
    with open(exp_dir + "payoffs", "rb") as f:
        payoffs = pickle.load(f)
    with open(baseline_dir + "partial_configs", "rb") as f:
        baseline_partial_configs = pickle.load(f)

    percent_opts = []
    fig1, ax1 = init_fig()
    legends = [
        "IQL",
        "IQ",
        "ModelS",
        "ModelR",
        "Hysteretic-Q",
        "Lenience",
        "Info-Q",
        "Info-Policy",
        "Comm-Bias",
    ]
    legends_swapped = [
        "IQL",
        "IQ",
        "ModelS",
        "ModelR",
        "Hysteretic-Q",
        "Lenience",
        "Comm-Bias",
        "Info-Q",
        "Info-Policy",
    ]
    for i in tqdm(range(len(partial_configs))):
        op = []
        for j in range(payoffs.shape[0]):
            with open(exp_dir + "output{}_{}".format(i, j), "rb") as f:
                op.append(pickle.load(f))
        rs_mean, rs_sem = split_mean(0, op, 100)
        xs = np.arange(len(rs_mean)) * 100
        ax1.plot(xs, rs_mean, color=color_sequence[i], label=legends[i])
        ax1.fill_between(
            xs, rs_mean - rs_sem, rs_mean + rs_sem, alpha=0.5, color=color_sequence[i]
        )
        # ax1.legend()

        percent_opt = [op[j][4] * 100 for j in range(len(op))]
        percent_opts.append(percent_opt)

    xs = [20000, 20000, 20000, 20000, 20000, 20000, 200, 200, 0]
    ys = [0.77, 0.73, 0.69, 0.65, 0.61, 0.57, 1, 0.95, 1.95]
    for i in range(len(legends)):
        color = color_sequence[i] if i != 5 else "#000000"
        ax1.text(xs[i], ys[i], legends[i], color=color)

    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Normalized reward")

    # fig1.savefig("norm_reward_32x32.pdf", format="pdf")

    # Swapping to keep comm-bias before info-Q
    po = percent_opts.pop(-1)
    percent_opts.insert(6, po)

    fig2, ax2 = init_fig()
    # ax2.spines['bottom'].set_visible(False)
    ax2.tick_params(axis="x", bottom=False, labelbottom=False)
    ax2.set_ylabel("% of runs converged to optimal policy")
    bplot = ax2.boxplot(percent_opts, notch=True, patch_artist=True)
    for patch, color in zip(bplot["boxes"], color_sequence_swapped):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    for i, alg in enumerate(legends_swapped):
        if i >= 7:
            ax2.text(i + 0.9, 105, alg, rotation=90, fontweight="bold")
        else:
            ax2.text(i + 0.9, 105, alg, rotation=90)

    # fig2.savefig("boxplot_32x32.pdf", format="pdf", bbox_inches="tight")


def split_mean(idx, op, bin_len=100):
    rs = np.array([op[j][idx] for j in range(len(op))])
    rs = np.dstack(np.split(rs, rs.shape[1] / bin_len, axis=1))
    rs = rs.mean(axis=1)
    rs_mean = rs.mean(axis=0)
    rs_sem = stats.sem(rs, axis=0)
    return rs_mean, rs_sem


if __name__ == "__main__":
    # random_payoffs("Experiments/PaperExperiments/{}/".format("1591032779"))
    # climbing_game()
    # cc_random_payoffs()

    plt.show()
