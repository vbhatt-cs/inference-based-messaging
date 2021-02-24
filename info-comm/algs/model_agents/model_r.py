import numpy as np

from ... import utils
from ..base import BaseAlg


class ModelR(BaseAlg):
    """
    The sender learns a model of the receiver and best responds to that model
    """

    def __init__(
        self,
        n_state,
        n_mess,
        n_act,
        n_runs,
        alpha0=0.1,
        eps0=0.1,
        eps0_decay=0,
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
            eps0 (float): Initial exploration rate
            eps0_decay (float): Decay of exploration rate per step
            mode (int): 0 for communication, 1 for fixed messages, 2 for fixed actions
        """
        self.n_state = n_state
        self.n_mess = n_mess
        self.n_act = n_act
        self.n_runs = n_runs
        self.alpha = alpha0
        self.eps = eps0
        self.eps_decay = eps0_decay
        self.mode = mode

        self.q0 = np.zeros((n_runs, n_state, n_mess))
        self.q1 = np.zeros((n_runs, n_mess, n_act))
        self.q = np.zeros((n_runs, n_state, n_act))

        self.s0 = None
        self.s1 = None
        self.m0 = None
        self.a1 = None
        self.a = None

    def act(self, state, test=False):
        """
        Args:
            state (np.ndarray): Current state (size=n_runs)
            test (bool): True if testing (no exploration)

        Returns:
            Current message, action (size=n_runs)
        """
        if test:  # Temporarily set eps to 0 to stop exploration
            true_eps = self.eps
            self.eps = 0

        self.s0 = state
        if self.mode == 1:
            self.m0 = self.s0
        else:
            self.m0 = np.zeros(self.n_runs, dtype=np.int)
            exp_mask = np.random.random(self.n_runs) < self.eps
            self.m0[exp_mask] = np.random.randint(self.n_mess, size=self.n_runs)[exp_mask]
            not_exp_mask = np.logical_not(exp_mask)
            a1_expected = np.zeros((self.n_runs, self.n_mess), dtype=np.int)
            exp_mask1 = np.random.random(self.n_runs) < self.eps
            a1_expected[exp_mask1] = np.random.randint(
                self.n_act, size=(self.n_runs, self.n_mess)
            )[exp_mask1]
            not_exp_mask1 = np.logical_not(exp_mask1)
            a1_expected[not_exp_mask1] = utils.rand_argmax(self.q1, axis=-1)[
                not_exp_mask1
            ]
            """
            Line after the comment is equivalent to the below code:
            q = np.zeros((self.n_runs, self.n_mess))
            for i in range(self.n_runs):
                q[i] = self.q[i, state[i]][a1_expected[i]]
            """
            q = self.q[
                np.arange(self.n_runs)[:, np.newaxis], state[:, np.newaxis], a1_expected
            ]
            self.m0[not_exp_mask] = utils.rand_argmax(q)[not_exp_mask]

        self.s1 = self.m0

        if self.mode == 0:
            self.a1 = a1_expected[np.arange(self.n_runs), self.s1]
        if self.mode == 2:
            self.a1 = self.s1
        else:
            self.a1 = np.zeros(self.n_runs, dtype=np.int)
            exp_mask = np.random.random(self.n_runs) < self.eps
            self.a1[exp_mask] = np.random.randint(self.n_act, size=self.n_runs)[exp_mask]
            not_exp_mask = np.logical_not(exp_mask)
            q = self.q1[np.arange(self.n_runs), self.s1]
            self.a1[not_exp_mask] = utils.rand_argmax(q)[not_exp_mask]

        if test:  # Revert value of eps
            self.eps = true_eps

        return self.m0, self.a1

    def train(self, reward):
        """
        Update the Q-values
        Args:
            reward (np.ndarray): Reward obtained (size=n_runs)
        """
        self.q[np.arange(self.n_runs), self.s0, self.a1] += self.alpha * (
            reward - self.q[np.arange(self.n_runs), self.s0, self.a1]
        )
        self.q1[np.arange(self.n_runs), self.s1, self.a1] += self.alpha * (
            reward - self.q1[np.arange(self.n_runs), self.s1, self.a1]
        )

        self.eps -= self.eps_decay

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
            for eps0 in [1, 0.5, 0.3, 0.1]:
                config = default_config.copy()
                config["alpha0"] = alpha0
                config["eps0"] = eps0
                config["eps0_decay"] = eps0 / (0.8 * default_config["episodes"])
                configs.append(config)
        return configs
