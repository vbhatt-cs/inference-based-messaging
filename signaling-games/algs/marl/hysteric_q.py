import numpy as np

from ... import utils
from ..base import BaseAlg


class HystericQ(BaseAlg):
    """
    Hysteric Q-Learning
    Matignon, Laëtitia, Guillaume J. Laurent, and Nadine Le Fort-Piat. "Hysteretic
    Q-Learning: An algorithm for decentralized reinforcement learning in cooperative
    multi-agent teams." 2007 IEEE/RSJ International Conference on Intelligent Robots and
    Systems. IEEE, 2007.
    """

    def __init__(
        self,
        n_state,
        n_mess,
        n_act,
        n_runs,
        alpha0=0.1,
        alpha1=0.01,
        eps0=0.1,
        eps0_decay=0,
        t0=0.1,
        t0_decay=0,
        mode=0,
        **kwargs
    ):
        """
        Args:
            n_state (int): Number of states
            n_mess (int): Number of messages
            n_act (int): Number of actions
            n_runs (int): Number of runs
            alpha0 (float): Step size for positive delta
            alpha1 (float): Step size for negative delta
            eps0 (float): Initial exploration rate
            eps0_decay (float): Decay of exploration rate per step
            t0 (float): Initial temperature
            t0_decay (float): Decay of temperature
            mode (int): 0 for communication, 1 for fixed messages, 2 for fixed actions
        """
        self.n_state = n_state
        self.n_mess = n_mess
        self.n_act = n_act
        self.n_runs = n_runs
        self.alpha = alpha0
        self.beta = alpha1
        self.eps = eps0
        self.eps_decay = eps0_decay
        self.t = t0
        self.t_decay = t0_decay
        self.mode = mode

        self.q0 = np.zeros((n_runs, n_state, n_mess))
        self.q1 = np.zeros((n_runs, n_mess, n_act))

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
            q = self.q0[np.arange(self.n_runs), state]
            if test:
                self.m0 = utils.rand_argmax(q)
            else:
                self.m0 = np.zeros(self.n_runs, dtype=np.int)
                exp_mask = np.random.random(self.n_runs) < self.eps
                self.m0[exp_mask] = np.random.randint(self.n_mess, size=self.n_runs)[
                    exp_mask
                ]
                not_exp_mask = np.logical_not(exp_mask)
                self.m0[not_exp_mask] = utils.rand_argmax(q)[not_exp_mask]

        self.s1 = self.m0

        if self.mode == 2:
            self.a1 = self.s1
        else:
            q = self.q1[np.arange(self.n_runs), self.s1]
            if test:
                self.a1 = utils.rand_argmax(q)
            else:
                self.a1 = np.zeros(self.n_runs, dtype=np.int)
                exp_mask = np.random.random(self.n_runs) < self.eps
                self.a1[exp_mask] = np.random.randint(self.n_act, size=self.n_runs)[
                    exp_mask
                ]
                not_exp_mask = np.logical_not(exp_mask)
                self.a1[not_exp_mask] = utils.rand_argmax(q)[not_exp_mask]

        return self.m0, self.a1

    def train(self, reward):
        """
        Update the Q-values
        Args:
            reward (np.ndarray): Reward obtained (size=n_runs)
        """
        delta0 = reward - self.q0[np.arange(self.n_runs), self.s0, self.m0]
        up0 = np.zeros_like(delta0)
        up0[delta0 < 0] = self.beta * delta0[delta0 < 0]
        up0[delta0 > 0] = self.alpha * delta0[delta0 > 0]
        self.q0[np.arange(self.n_runs), self.s0, self.m0] += up0
        delta1 = reward - self.q1[np.arange(self.n_runs), self.s1, self.a1]
        up1 = np.zeros_like(delta1)
        up1[delta1 < 0] = self.beta * delta1[delta1 < 0]
        up1[delta1 > 0] = self.alpha * delta1[delta1 > 0]
        self.q1[np.arange(self.n_runs), self.s1, self.a1] += up1

        self.eps -= self.eps_decay
        self.t *= self.t_decay

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
            for alpha1_mult in [0.1, 1, 10]:
                for eps0 in [1, 0.5, 0.3, 0.1]:
                    config = default_config.copy()
                    config["alpha0"] = alpha0
                    config["alpha1"] = alpha0 * alpha1_mult
                    config["eps0"] = eps0
                    config["eps0_decay"] = eps0 / (0.8 * default_config["episodes"])
                    configs.append(config)
        return configs
