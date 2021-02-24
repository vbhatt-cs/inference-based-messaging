import numpy as np

from ... import utils
from ..base import BaseAlg


class Lenience(BaseAlg):
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

        self.q0 = np.zeros((n_runs, n_state, n_mess)) - 1
        self.t0 = np.zeros((n_runs, n_state, n_mess)) + self.max_temp
        self.q1 = np.zeros((n_runs, n_mess, n_act)) - 1
        self.t1 = np.zeros((n_runs, n_mess, n_act)) + self.max_temp

        self.s0 = None
        self.s1 = None
        self.m0 = None
        self.a1 = None

    def act(self, state, test=False):
        """
        Args:
            state (np.ndarray): Current state (size=n_runs)
            test (bool): True if testing (no exploration)

        Returns:
            Current message, action (size=n_runs)
        """
        self.s0 = state
        if self.mode == 1:
            self.m0 = self.s0
        else:
            q = self.q0[np.arange(self.n_runs), self.s0]
            if test:
                self.m0 = utils.rand_argmax(q)
            else:
                self.m0 = np.zeros(self.n_runs, dtype=np.int)
                t_s = self.t0[np.arange(self.n_runs), self.s0].mean(axis=-1)
                greedy_mask = t_s < self.min_temp
                self.m0[greedy_mask] = utils.rand_argmax(q)[greedy_mask]
                q -= q.max(axis=1, keepdims=True)
                q_exp = np.exp(q / (self.omega * t_s[:, np.newaxis]))
                p = q_exp / q_exp.sum(axis=1, keepdims=True)
                self.m0[~greedy_mask] = utils.vectorized_2d_choice(
                    np.arange(self.n_mess), p=p
                )[~greedy_mask]

        self.s1 = self.m0

        if self.mode == 2:
            self.a1 = self.s1
        else:
            q = self.q1[np.arange(self.n_runs), self.s1]
            if test:
                self.a1 = utils.rand_argmax(q)
            else:
                self.a1 = np.zeros(self.n_runs, dtype=np.int)
                t_s = self.t1[np.arange(self.n_runs), self.s1].mean(axis=-1)
                greedy_mask = t_s < self.min_temp
                self.a1[greedy_mask] = utils.rand_argmax(q)[greedy_mask]
                q -= q.max(axis=1, keepdims=True)
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
            self.q0[np.arange(self.n_runs), self.s0, self.m0] <= reward,
            rand
            < (
                1
                - np.exp(
                    -1 / self.theta * self.t0[np.arange(self.n_runs), self.s0, self.m0]
                )
            ),
        )
        self.q0[update_mask, self.s0[update_mask], self.m0[update_mask]] += (
            self.alpha
            * (reward - self.q0[np.arange(self.n_runs), self.s0, self.m0])[update_mask]
        )
        self.t0[np.arange(self.n_runs), self.s0, self.m0] *= self.delta

        rand = np.random.random(self.n_runs)
        update_mask = np.logical_or(
            self.q0[np.arange(self.n_runs), self.s0, self.m0] <= reward,
            rand
            < (
                1
                - np.exp(
                    -1 / self.theta * self.t0[np.arange(self.n_runs), self.s0, self.m0]
                )
            ),
        )
        self.q1[update_mask, self.s1[update_mask], self.a1[update_mask]] += (
            self.alpha
            * (reward - self.q1[np.arange(self.n_runs), self.s1, self.a1])[update_mask]
        )
        self.t1[np.arange(self.n_runs), self.s1, self.a1] *= self.delta

    @staticmethod
    def generate_configs(default_config):
        """
        An optional method that generates the list of configs for hyperparameter tuning
        Args:
            default_config (dict): Default values of the parameters
        Returns:
            List of dicts with the required parameters
        """
        configs = []
        for alpha0 in [0.01, 0.05, 0.1, 0.5]:
            for delta0 in [0.999, 0.995, 0.99]:
                for max_temp0 in [5, 50, 500, 5000]:
                    for omega0 in [0.1, 1, 10]:
                        for theta0 in [0.1, 1, 10]:
                            config = default_config.copy()
                            config["alpha0"] = alpha0
                            config["delta0"] = delta0
                            config["max_temp0"] = max_temp0
                            config["min_temp0"] = max_temp0 * (
                                delta0 ** (0.8 * default_config["episodes"])
                            )
                            config["omega0"] = omega0
                            config["theta0"] = theta0
                            configs.append(config)
        return configs
