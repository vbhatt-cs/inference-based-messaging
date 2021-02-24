import numpy as np

from ... import utils
from ..base import BaseAlg


class InfoPolicy(BaseAlg):
    """
    Inference based messaging with the receiver using REINFORCE
    """

    def __init__(
        self,
        n_state,
        n_mess,
        n_act,
        n_runs,
        alpha0=0.1,
        alpha_critic0=0.1,
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
            alpha_critic0 (float): Critic step size
            mode (int): 0 for communication, 1 for fixed messages, 2 for fixed actions
        """
        self.n_state = n_state
        self.n_mess = n_mess
        self.n_act = n_act
        self.n_runs = n_runs
        self.alpha = alpha0
        self.alpha_critic = alpha_critic0
        self.mode = mode

        self.h1 = np.zeros((n_runs, n_mess, n_act))
        self.v1 = np.zeros((n_runs, n_mess))

        self.q0 = np.zeros((n_runs, n_state, n_mess)) - 2

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
            a0_expected = utils.rand_argmax(self.q0, axis=-1)
            p_ms = np.zeros((self.n_runs, self.n_mess))
            p_sm = np.zeros((self.n_runs, self.n_mess))
            for m in range(self.n_mess):
                """ 
                Code below the comment is equivalent to
                for i in range(self.n_runs):
                    p_ms = a0_expected[i, self.s0[i]] == m
                    p_m = (a0_expected[i] == m).sum()
                    if p_m != 0:
                        p_sm = p_ms / p_m
                    else:
                        p_sm = 1 / self.n_state
                """
                p_ms[:, m] = a0_expected[np.arange(self.n_runs), self.s0] == m
                p_m = (a0_expected == m).sum(axis=1)
                non_zero = p_m != 0
                p_sm[non_zero, m] = (p_ms[:, m] / p_m)[non_zero]
                p_sm[np.logical_not(non_zero), m] = 1

            self.m0 = utils.rand_argmax(p_sm)

        self.s1 = self.m0

        if self.mode == 2:
            self.a1 = self.s1
        else:
            h1_s = self.h1[np.arange(self.n_runs), self.s1]
            pi_s = utils.softmax(h1_s, axis=1)
            self.a1 = utils.vectorized_2d_choice(np.arange(self.n_act), p=pi_s)

        return self.m0, self.a1

    def train(self, reward):
        """
        Update the Q-values
        Args:
            reward (np.ndarray): Reward obtained (size=n_runs)
        """
        self.q0[np.arange(self.n_runs), self.s0, self.m0] += (
            self.alpha / 10 * (reward - self.q0[np.arange(self.n_runs), self.s0, self.m0])
        )

        delta1 = reward - self.v1[np.arange(self.n_runs), self.s1]
        # delta1 = reward
        self.v1[np.arange(self.n_runs), self.s1] += self.alpha_critic * delta1
        h1_s = self.h1[np.arange(self.n_runs), self.s1]
        grad = -utils.softmax(h1_s, axis=1)
        grad[np.arange(self.n_runs), self.a1] += 1
        self.h1[np.arange(self.n_runs), self.s1] += (
            self.alpha * delta1[:, np.newaxis] * grad
        )

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
        for alpha in [0.1, 0.5]:
            config = default_config.copy()
            config["alpha0"] = alpha
            config["alpha_critic0"] = alpha
            configs.append(config)
        return configs
