import numpy as np

from ... import utils
from ..base import BaseAlg


class InfoQ(BaseAlg):
    """
    Inference based messaging with the receiver using Q-Learning
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
        q_mod=False,
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
            q_mod (bool): True if initial q values should be modified, False otherwise
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

        if q_mod:
            self.q0 -= 2
            self.q1 += 2

        self.s0 = None
        self.s1 = None
        self.m0 = None
        self.a1 = None
        self.n_s = np.zeros((self.n_runs, self.n_state))

    def act(self, state, test=False):
        """
        Args:
            state (np.ndarray): Current state (size=n_runs)
            test (bool): True if testing (no exploration)

        Returns:
            Current message, action (size=n_runs)
        """
        self.s0 = state
        self.n_s[np.arange(self.n_runs), self.s0] += 1
        if self.mode == 1:
            self.m0 = self.s0
        else:
            a0_expected = utils.rand_argmax(self.q0, axis=-1)
            p_ms = np.zeros((self.n_runs, self.n_mess))
            p_sm = np.zeros((self.n_runs, self.n_mess))
            p_s = self.n_s / self.n_s.sum(axis=1, keepdims=True)
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
                p_m = ((a0_expected == m) * p_s).sum(axis=1)
                non_zero = p_m != 0
                p_sm[non_zero, m] = (
                    p_ms[:, m] * p_s[np.arange(self.n_runs), self.s0] / p_m
                )[non_zero]
                p_sm[np.logical_not(non_zero), m] = 1

            self.m0 = utils.rand_argmax(p_sm)

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
        self.q0[np.arange(self.n_runs), self.s0, self.m0] += self.alpha * (
            reward - self.q0[np.arange(self.n_runs), self.s0, self.m0]
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
        config = default_config.copy()
        config["alpha0"] = 0.1
        config["eps0"] = 0
        config["eps0_decay"] = 0
        return [config]
