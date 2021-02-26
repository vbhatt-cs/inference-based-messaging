import numpy as np

from ... import utils
from ..base import BaseAlg


class IQL(BaseAlg):
    """
    Independent Q-Learning.
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
        configs = []
        for alpha0 in [0.01, 0.05, 0.1, 0.5]:
            for eps0 in [1, 0.5, 0.3, 0.1]:
                config = default_config.copy()
                config["alpha0"] = alpha0
                config["eps0"] = eps0
                config["eps0_decay"] = eps0 / (0.8 * default_config["episodes"])
                configs.append(config)
        return configs


class IQ(BaseAlg):
    """
    Iterative Learning. One agent is trained while the other is fixed and vice versa.
    """

    def __init__(
        self,
        n_state,
        n_mess,
        n_act,
        n_runs,
        alpha0=0.1,
        alpha1=0.1,
        eps0=0.1,
        eps0_decay=0,
        eps1=0.1,
        eps1_decay=0,
        period=1,
        mode=0,
        **kwargs
    ):
        """
        Args:
            n_state (int): Number of states
            n_mess (int): Number of messages
            n_act (int): Number of actions
            n_runs (int): Number of runs
            alpha0 (float): Step size for agent 1
            alpha1 (float): Step size for agent 2
            eps0 (float): Exploration rate for agent 1 at the beginning of its training
                cycle
            eps0_decay (float): Decay of exploration rate per step for agent 1
            eps1 (float): Exploration rate for agent 2 at the beginning of its training
                cycle
            eps1_decay (float): Decay of exploration rate per step for agent 2
            period (int): Period after which to switch training
            mode (int): 0 for communication, 1 for fixed messages, 2 for fixed actions
        """
        self.n_state = n_state
        self.n_mess = n_mess
        self.n_act = n_act
        self.n_runs = n_runs
        self.mode = mode

        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.eps0_max = eps0
        self.eps0 = eps0
        self.eps0_decay = eps0_decay
        self.eps1_max = eps1
        self.eps1 = eps1
        self.eps1_decay = eps1_decay
        self.period = period

        self.q0 = np.zeros((n_runs, n_state, n_mess))
        self.q1 = np.zeros((n_runs, n_mess, n_act))

        self.s0 = None
        self.s1 = None
        self.m0 = None
        self.a1 = None
        self.step = 0
        self.cycle = False

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
                exp_mask = np.random.random(self.n_runs) < self.eps0
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
                exp_mask = np.random.random(self.n_runs) < self.eps1
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
        # Every 'period' number of steps, switch the agent to train and set the
        # exploration rate back to maximum
        if self.step == self.period:
            self.cycle = not self.cycle
            self.step = 0
            self.eps0 = self.eps0_max
            self.eps1 = self.eps1_max

        if self.cycle:
            self.eps1 = 0
            self.q0[np.arange(self.n_runs), self.s0, self.m0] += self.alpha0 * (
                reward - self.q0[np.arange(self.n_runs), self.s0, self.m0]
            )
            self.eps0 -= self.eps0_decay

        else:
            self.eps0 = 0
            self.q1[np.arange(self.n_runs), self.s1, self.a1] += self.alpha1 * (
                reward - self.q1[np.arange(self.n_runs), self.s1, self.a1]
            )
            self.eps1 -= self.eps1_decay

        self.step += 1

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
                    for period in [1, 10, 100]:
                        config = default_config.copy()
                        config["alpha0"] = alpha0
                        config["alpha1"] = alpha0 * alpha1_mult
                        config["eps0"] = eps0
                        config["eps0_decay"] = eps0 / (0.8 * period)
                        config["eps1"] = eps0
                        config["eps1_decay"] = eps0 / (0.8 * period)
                        config["period"] = period
                        configs.append(config)
        return configs


class CentralizedQLearning(BaseAlg):
    """
    Centralized Q-Learning
    """

    def __init__(
        self, n_state, n_mess, n_act, n_runs, alpha0=0.1, eps0=0.1, eps0_decay=0, **kwargs
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
        """
        self.n_state = n_state
        self.n_mess = n_mess
        self.n_act = n_act
        self.n_total_act = n_mess * n_act
        self.n_runs = n_runs
        self.alpha = alpha0
        self.eps = eps0
        self.eps_decay = eps0_decay

        self.q = np.zeros((n_runs, n_state, self.n_total_act))

        self.s0 = None
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
        self.s0 = state
        q = self.q[np.arange(self.n_runs), state]
        if test:
            self.a = utils.rand_argmax(q)
        else:
            self.a = np.zeros(self.n_runs, dtype=np.int)
            exp_mask = np.random.random(self.n_runs) < self.eps
            self.a[exp_mask] = np.random.randint(self.n_total_act, size=self.n_runs)[
                exp_mask
            ]
            not_exp_mask = np.logical_not(exp_mask)
            self.a[not_exp_mask] = utils.rand_argmax(q)[not_exp_mask]

        self.m0 = self.a % self.n_mess
        self.a1 = self.a // self.n_mess

        return self.m0, self.a1

    def train(self, reward):
        """
        Update the Q-values
        Args:
            reward (np.ndarray): Reward obtained (size=n_runs)
        """
        self.q[np.arange(self.n_runs), self.s0, self.a] += self.alpha * (
            reward - self.q[np.arange(self.n_runs), self.s0, self.a]
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
