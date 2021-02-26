import argparse
import os

import numpy as np
from scipy.stats import stats

import algs

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description="Communication in Matrix Games")
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        metavar="N",
        help="number of episodes (default: 10000)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1000,
        metavar="N",
        help="number of runs to repeat (default: 1000)",
    )

    parser.add_argument(
        "--payoff",
        nargs="*",
        type=int,
        default=[11, -30, 0, -30, 7, 6, 0, 0, 5],
        metavar="R",
        help="payoff matrix values (space separated integers)",
    )
    parser.add_argument(
        "--payoff-sigma",
        type=float,
        default=1,
        metavar="R",
        help="standard deviation (relative to mean) of payoff (default: 1)",
    )
    parser.add_argument(
        "--states", type=int, default=3, metavar="S", help="number of states (default: 3)"
    )
    parser.add_argument(
        "--actions",
        type=int,
        default=3,
        metavar="A",
        help="number of actions (default: 3)",
    )
    parser.add_argument(
        "--messages",
        type=int,
        default=3,
        metavar="M",
        help="number of messages (default: 3)",
    )
    parser.add_argument(
        "--alpha0",
        type=float,
        default=0.1,
        metavar="LR",
        help="step size for agent 1 (default: 0.1)",
    )
    parser.add_argument(
        "--eps0",
        type=float,
        default=0.1,
        metavar="E",
        help="initial exploration rate for agent 1 (default: 0.1)",
    )
    parser.add_argument(
        "--eps0-decay",
        type=float,
        default=0.1 / 8000,
        metavar="ED",
        help="decay amount of exploration rate for agent 1 (default: 0.1 / 8000)",
    )
    parser.add_argument(
        "--alpha1",
        type=float,
        default=0.1,
        metavar="LR",
        help="step size for agent 2 (default: 0.1)",
    )
    parser.add_argument(
        "--eps1",
        type=float,
        default=0.1,
        metavar="E",
        help="initial exploration rate for agent 2 (default: 0.1)",
    )
    parser.add_argument(
        "--eps1-decay",
        type=float,
        default=0.1 / 8000,
        metavar="ED",
        help="decay amount of exploration rate for agent 2 (default: 0.1 / 8000)",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=10000,
        metavar="N",
        help="number of episodes before switch for IQ (default: 10000)",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="IQL",
        metavar="A",
        choices=["IQL", "IQ", "ModelR", "ModelS"],
        help="algorithm to use: see algs/__init__.py for list of algorithms (default: 'IQL')",
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        metavar="M",
        choices=[0, 1, 2],
        help="mode:\n"
        "0 - Standard communication\n"
        "1 - Identity policy for agent 1's messages\n"
        "2 - Identity policy for agent 2's actions (default: 0)",
    )

    """Optional args for certain algorithms"""
    # ModelS
    parser.add_argument(
        "--act-variant",
        type=str,
        default="exact",
        metavar="A",
        choices=["exact", "q0_imperfect", "seed_imperfect", "imperfect"],
        help="optional arg for ModelS: see algs/model_agents/model_s.py for documentation "
        "(default: 'exact')",
    )
    parser.add_argument(
        "--train-variant",
        type=str,
        default="exact",
        metavar="A",
        choices=["exact", "imperfect"],
        help="optional arg for ModelS: see algs/model_agents/model_s.py for documentation "
        "(default: 'exact')",
    )
    parser.add_argument(
        "--sample-variant",
        type=str,
        default="sample",
        metavar="A",
        choices=["sample", "expectation"],
        help="optional arg for ModelS: see algs/model_agents/model_s.py for documentation "
        "(default: 'exact')",
    )

    # Boltzmann exploration
    parser.add_argument(
        "--t0",
        type=float,
        default=100,
        metavar="E",
        help="initial temperature for agent 1 (default: 0.1)",
    )
    parser.add_argument(
        "--t0-decay",
        type=float,
        default=0.9,
        metavar="ED",
        help="temperature decay for agent 1 (default: 0.9)",
    )

    # Information maximization
    parser.add_argument(
        "--q-mod",
        action="store_true",
        default=False,
        help="modifies initial q values to help convergence",
    )

    # PSO
    parser.add_argument(
        "--n-pop",
        type=int,
        default=10,
        metavar="N",
        help="number of particles (default: 10)",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=0.99,
        metavar="LR",
        help="omega for PSO (default: 0.99)",
    )
    parser.add_argument(
        "--phi-p", type=float, default=2, metavar="LR", help="phi_p for PSO (default: 2)"
    )
    parser.add_argument(
        "--phi-g", type=float, default=2, metavar="LR", help="phi_g for PSO (default: 2)"
    )

    # Policy gradient
    parser.add_argument(
        "--alpha-critic0",
        type=float,
        default=0.1,
        metavar="LR",
        help="step size for critic of agent 1 (default: 0.1)",
    )
    parser.add_argument(
        "--beta-reg0",
        type=float,
        default=0.1,
        metavar="LR",
        help="weightage given to regularization penalty (default: 0.1)",
    )

    # Lenience
    parser.add_argument(
        "--delta0",
        type=float,
        default=0.995,
        metavar="ED",
        help="temperature decay parameter (default: 0.995)",
    )
    parser.add_argument(
        "--max-temp0",
        type=float,
        default=50.0,
        metavar="E",
        help="maximum temperature (default: 50.0)",
    )
    parser.add_argument(
        "--min_temp0",
        type=float,
        default=2.0,
        metavar="E",
        help="minimum temperature (default: 2.0)",
    )
    parser.add_argument(
        "--omega0",
        type=float,
        default=1.0,
        metavar="LR",
        help="action selection moderation factor (default: 1.0)",
    )
    parser.add_argument(
        "--theta0",
        type=float,
        default=1.0,
        metavar="LR",
        help="lenience moderation factor (default: 1.0)",
    )

    # Sampling
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        metavar="N",
        help="number of samples for estimation (default: 10)",
    )
    parser.add_argument(
        "--zero-correction",
        action="store_true",
        default=False,
        help="if true, p(s|m) is set to slightly lower than 1 instead of exactly 1 when p(m) = 0",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="number of episodes between each logging (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, metavar="S", help="random seed (default: 0)"
    )
    return parser.parse_args()


def run(config):
    np.random.seed(config["seed"])
    try:
        payoff = (
            np.array(config["payoff"])
            .reshape((config["states"], config["actions"]))
            .astype(np.float)
        )
        payoff /= np.abs(payoff).max()
    except ValueError:
        raise ValueError(
            "There should be {} (states * actions) elements in payoff. Found {} elements".format(
                config["states"] * config["actions"], len(config["payoff"])
            )
        )

    # Logging vars
    rewards = np.zeros((config["episodes"] // config["log_interval"], config["runs"]))
    opt = np.zeros((config["episodes"] // config["log_interval"], config["runs"]))
    converge_point = np.zeros((config["states"], config["actions"]))

    alg = getattr(algs, config["algorithm"])(
        config["states"], config["messages"], config["actions"], config["runs"], **config
    )

    # Train the algorithms
    for e in range(config["episodes"]):
        s0 = np.random.randint(config["states"], size=config["runs"])
        # s0 = np.random.choice(config['states'], size=config['runs'], p=[0.1, 0.8, 0.1])
        m0, a1 = alg.act(s0)
        r_mean = payoff[s0, a1]
        r = r_mean * (1 + config["payoff_sigma"] * np.random.randn(config["runs"]))
        # r = r_mean * (2 * np.random.randint(2, size=config['runs']))
        # regret[e // config["log_interval"]] += (
        #     (payoff[s0].max(axis=1) - r_mean)
        #     / (payoff[s0].max(axis=1) - payoff[s0].min(axis=1))
        #     / config["log_interval"]
        # )
        rewards[e // config["log_interval"]] += (
            r_mean / payoff[s0].max(axis=1) / config["log_interval"]
        )
        # rewards[e // config['log_interval']] += r_mean / config['log_interval']
        opt[e // config["log_interval"]] += (
            np.isclose(r_mean, payoff[s0].max(axis=1)) / config["log_interval"]
        )
        alg.train(r)

    # Evaluate using a state sweep
    message_policy = np.zeros((config["runs"], config["states"]))
    percent_opt = 0
    for s in range(config["states"]):
        s0 = s + np.zeros(config["runs"], dtype=np.int)
        m0, a1 = alg.act(s0, test=True)
        message_policy[:, s] = m0
        converge_point[s] = np.bincount(a1, minlength=config["actions"])
        best_act = payoff[s].argmax()
        percent_opt += converge_point[s, best_act] / config["runs"] / config["states"]

    if config["log_summary"]:
        return (
            rewards.mean(axis=1),
            stats.sem(rewards, axis=1),
            opt.mean(axis=1),
            stats.sem(opt, axis=1),
            percent_opt,
            converge_point,
        )

    # Calculate the numbers corresponding to uniqueness of message protocol
    n_diff = np.count_nonzero(
        np.logical_and(
            np.logical_and(
                message_policy[:, 0] != message_policy[:, 1],
                message_policy[:, 1] != message_policy[:, 2],
            ),
            message_policy[:, 0] != message_policy[:, 2],
        )
    )
    n01 = np.count_nonzero(message_policy[:, 0] == message_policy[:, 1])
    n12 = np.count_nonzero(message_policy[:, 1] == message_policy[:, 2])
    n02 = np.count_nonzero(message_policy[:, 0] == message_policy[:, 2])
    n012 = np.count_nonzero(
        np.logical_and(
            message_policy[:, 0] == message_policy[:, 1],
            message_policy[:, 1] == message_policy[:, 2],
        )
    )
    ns = (n_diff, n01, n12, n02, n012)

    return opt, converge_point, ns


if __name__ == "__main__":
    args = parse_args()
    run(vars(args))
