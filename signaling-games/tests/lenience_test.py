import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import stats
from tqdm import tqdm

from .. import plot_utils
from .. import utils

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


class Lenience:
    """
    Lenient learners.
    Wei, Ermo, and Sean Luke. "Lenient learning in independent-learner stochastic
    cooperative games." The Journal of Machine Learning Research 17.1 (2016): 2914-2955.
    """

    def __init__(
        self,
        n_state,
        n_mess,
        n_act,
        n_runs,
        alpha0=0.1,
        delta0=0.995,
        max_temp0=50.0,
        min_temp0=2.0,
        omega0=1.0,
        theta0=1.0,
        mode=0,
        **kwargs
    ):
        """
        Args:
            n_state (int): Number of states
            n_mess (int): Number of messages
            n_act (int): Number of actions
            n_runs (int): Number of runs
            alpha0 (float): Step size
            delta0 (float): Temperature decay parameter
            max_temp0 (float): Maximum temperature
            min_temp0 (float): Minimum temperature
            omega0 (float): Action selection moderation factor
            theta0 (float): Lenience moderation factor
            mode (int): 0 for communication, 1 for fixed messages, 2 for fixed actions
        """
        self.n_state = n_state
        self.n_mess = n_mess
        self.n_act = n_act
        self.n_runs = n_runs
        self.alpha = alpha0
        self.delta = delta0
        self.max_temp = max_temp0
        self.min_temp = min_temp0
        self.omega = omega0
        self.theta = theta0
        self.mode = mode

        self.q0 = np.zeros((n_runs, n_mess)) - 1
        self.t0 = np.zeros((n_runs, n_mess)) + self.max_temp
        self.q1 = np.zeros((n_runs, n_act)) - 1
        self.t1 = np.zeros((n_runs, n_act)) + self.max_temp

        self.s0 = None
        self.s1 = None
        self.m0 = None
        self.a1 = None

    def act(self, test=False):
        """
        Args:
            test (bool): True if testing (no exploration)

        Returns:
            Current message, action (size=n_runs)
        """
        if test:
            self.m0 = utils.rand_argmax(self.q0)
        else:
            self.m0 = np.zeros(self.n_runs, dtype=np.int)
            t_s = self.t0.mean(axis=-1)
            greedy_mask = t_s < self.min_temp
            self.m0[greedy_mask] = utils.rand_argmax(self.q0)[greedy_mask]
            q = self.q0 - self.q0.max(axis=1, keepdims=True)
            q_exp = np.exp(q / (self.omega * t_s[:, np.newaxis]))
            p = q_exp / q_exp.sum(axis=1, keepdims=True)
            self.m0[~greedy_mask] = utils.vectorized_2d_choice(
                np.arange(self.n_mess), p=p
            )[~greedy_mask]

        if test:
            self.a1 = utils.rand_argmax(self.q1)
        else:
            self.a1 = np.zeros(self.n_runs, dtype=np.int)
            t_s = self.t1.mean(axis=-1)
            greedy_mask = t_s < self.min_temp
            self.a1[greedy_mask] = utils.rand_argmax(self.q1)[greedy_mask]
            q = self.q1 - self.q1.max(axis=1, keepdims=True)
            q_exp = np.exp(q / (self.omega * t_s[:, np.newaxis]))
            p = q_exp / q_exp.sum(axis=1, keepdims=True)
            self.a1[~greedy_mask] = utils.vectorized_2d_choice(
                np.arange(self.n_mess), p=p
            )[~greedy_mask]

        return self.m0, self.a1

    def train(self, reward):
        """
        Update the Q-values
        Args:
            reward (np.ndarray): Reward obtained (size=n_runs)
        """
        rand = np.random.random(self.n_runs)
        update_mask = np.logical_or(
            self.q0[np.arange(self.n_runs), self.m0] <= reward,
            rand
            < (1 - np.exp(-1 / self.theta * self.t0[np.arange(self.n_runs), self.m0])),
        )
        self.q0[update_mask, self.m0[update_mask]] += (
            self.alpha * (reward - self.q0[np.arange(self.n_runs), self.m0])[update_mask]
        )
        self.t0[np.arange(self.n_runs), self.m0] *= self.delta

        rand = np.random.random(self.n_runs)
        update_mask = np.logical_or(
            self.q0[np.arange(self.n_runs), self.m0] <= reward,
            rand
            < (1 - np.exp(-1 / self.theta * self.t0[np.arange(self.n_runs), self.m0])),
        )
        self.q1[update_mask, self.a1[update_mask]] += (
            self.alpha * (reward - self.q1[np.arange(self.n_runs), self.a1])[update_mask]
        )
        self.t1[np.arange(self.n_runs), self.a1] *= self.delta


def run(config):
    payoff = np.array([11, -30, 0, -30, 7, 6, 0, 0, 5.0]).reshape((3, 3))
    # payoff = np.array([10, 0, 0, 0, 2, 0, 0, 0, 10.]).reshape((3, 3))
    payoff /= np.abs(payoff).max()
    alg = Lenience(
        config["states"], config["messages"], config["actions"], config["runs"], **config
    )
    rewards = np.zeros((config["episodes"] // config["log_interval"], config["runs"]))
    converge_point = np.zeros((config["states"], config["actions"]))

    for e in tqdm(range(config["episodes"])):
        m0, a1 = alg.act()
        r = payoff[m0, a1]
        alg.train(r)
        rewards[e // config["log_interval"]] += r / config["log_interval"]

    # Evaluate using a state sweep
    m0, a1 = alg.act(test=True)
    for i in range(config["runs"]):
        converge_point[m0[i], a1[i]] += 1

    return rewards, converge_point


def main():
    with open("../default_config.json") as f:
        default_config = json.load(f)

    rs, converge_point = run(default_config)

    fig1, ax1 = plt.subplots()
    # ax1.set_ylim(-0.2, 1.05)
    ax1.set_ylabel("Obtained reward")
    ax1.set_xlabel("Episodes")
    ys = rs.mean(axis=1)
    rs_sem = stats.sem(rs, axis=1)
    xs = np.arange(len(ys)) * default_config["log_interval"]
    ax1.plot(xs, ys, color=color_sequence[0])
    plt.fill_between(xs, ys - rs_sem, ys + rs_sem, alpha=0.5, color=color_sequence[0])

    # Plot the convergence points and save it
    fig2, ax2 = plot_utils.plot_confusion_matrix(converge_point.astype(np.int))
    plt.show()


main()
