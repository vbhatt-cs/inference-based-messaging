import numpy as np

from ... import utils
from ..base import BaseAlg


class ModelS(BaseAlg):
    """
    The receiver learns a model of the sender and uses it to predict the state given the
    message
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
        act_variant="exact",
        train_variant="exact",
        sample_variant="sample",
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
            act_variant (str):
                'exact' for exact model of q0, common seed;
                'q0_imperfect' for imperfect model of q0, common seed;
                'seed_imperfect' for exact model of q0, different seed;
                'imperfect' for imperfect model of q0, different seed (default: 'exact')
            train_variant (str):
                'exact' for using true state to train q
                'imperfect' for using sampled state to train q (default: 'exact')
            sample_variant (str):
                'sample' for sampling the state
                'expectation' for taking expected Q values of all possible states
                    (default: 'sample')
        """
        self.n_state = n_state
        self.n_mess = n_mess
        self.n_act = n_act
        self.n_runs = n_runs
        self.alpha = alpha0
        self.eps = eps0
        self.eps_decay = eps0_decay
        self.mode = mode
        self.act_variant = act_variant
        self.train_variant = train_variant
        self.sample_variant = sample_variant

        self.q0 = np.zeros((n_runs, n_state, n_mess))
        self.q1 = np.zeros((n_runs, n_mess, n_act))
        self.q = np.zeros((n_runs, n_state, n_act))
        self.q0_expected = np.zeros((n_runs, n_state, n_mess))

        self.s0 = None
        self.s1 = None
        self.m0 = None
        self.a1 = None
        self.s0_sampled = None
        self.p_s0_possible = None

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
            a0_expected = np.zeros((self.n_runs, self.n_state), dtype=np.int)
            a0_expected[:] = np.arange(self.n_state)
            self.m0 = self.s0
        else:
            a0_expected = np.zeros((self.n_runs, self.n_state), dtype=np.int)
            exp_mask = np.random.random(self.n_runs) < self.eps
            a0_expected[exp_mask] = np.random.randint(
                self.n_mess, size=(self.n_runs, self.n_state)
            )[exp_mask]
            not_exp_mask = np.logical_not(exp_mask)

            if self.act_variant == "exact":
                a0_expected[not_exp_mask] = utils.rand_argmax(self.q0, axis=-1)[
                    not_exp_mask
                ]
                self.m0 = a0_expected[np.arange(self.n_runs), self.s0]
            elif self.act_variant == "q0_imperfect":
                a0_expected[not_exp_mask] = utils.rand_argmax(self.q0_expected, axis=-1)[
                    not_exp_mask
                ]
                self.m0 = np.zeros(self.n_runs, dtype=np.int)
                self.m0[exp_mask] = a0_expected[np.arange(self.n_runs), self.s0][exp_mask]
                q = self.q0[np.arange(self.n_runs), state]
                self.m0[not_exp_mask] = utils.rand_argmax(q)[not_exp_mask]
            elif self.act_variant == "seed_imperfect":
                a0_expected[not_exp_mask] = utils.rand_argmax(self.q0, axis=-1)[
                    not_exp_mask
                ]
                self.m0 = np.zeros(self.n_runs, dtype=np.int)
                exp_mask = np.random.random(self.n_runs) < self.eps
                self.m0[exp_mask] = np.random.randint(self.n_mess, size=self.n_runs)[
                    exp_mask
                ]
                not_exp_mask = np.logical_not(exp_mask)
                q = self.q0[np.arange(self.n_runs), state]
                self.m0[not_exp_mask] = utils.rand_argmax(q)[not_exp_mask]
            elif self.act_variant == "imperfect":
                a0_expected[not_exp_mask] = utils.rand_argmax(self.q0_expected, axis=-1)[
                    not_exp_mask
                ]
                self.m0 = np.zeros(self.n_runs, dtype=np.int)
                exp_mask = np.random.random(self.n_runs) < self.eps
                self.m0[exp_mask] = np.random.randint(self.n_mess, size=self.n_runs)[
                    exp_mask
                ]
                not_exp_mask = np.logical_not(exp_mask)
                q = self.q0[np.arange(self.n_runs), state]
                self.m0[not_exp_mask] = utils.rand_argmax(q)[not_exp_mask]
            else:
                raise ValueError("Incorrect act variant")

        self.s1 = self.m0

        if self.mode == 2:
            self.p_s0_possible = np.zeros((self.n_runs, self.n_state))
            self.p_s0_possible[np.arange(self.n_runs), self.s0] = 1
            self.s0_sampled = self.s0
            self.a1 = self.s1
        else:
            s0_possible = a0_expected == self.s1[:, np.newaxis]
            # If agent 2 thinks that agent 1 can't send the message from any state, give
            # equal probability to all states
            s0_possible[s0_possible.sum(axis=1) == 0] = 1
            self.p_s0_possible = s0_possible / s0_possible.sum(axis=1, keepdims=True)
            self.s0_sampled = utils.vectorized_2d_choice(
                np.arange(self.n_state), p=self.p_s0_possible
            )
            self.a1 = np.zeros(self.n_runs, dtype=np.int)
            exp_mask = np.random.random(self.n_runs) < self.eps
            self.a1[exp_mask] = np.random.randint(self.n_act, size=self.n_runs)[exp_mask]
            not_exp_mask = np.logical_not(exp_mask)
            if self.sample_variant == "sample":
                q = self.q[np.arange(self.n_runs), self.s0_sampled]
            elif self.sample_variant == "expectation":
                q = np.einsum("ijk,ij->ik", self.q, self.p_s0_possible)
            else:
                raise ValueError("Incorrect sample variant")
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
        if self.train_variant == "exact":
            self.q[np.arange(self.n_runs), self.s0, self.a1] += self.alpha * (
                reward - self.q[np.arange(self.n_runs), self.s0, self.a1]
            )
        elif self.train_variant == "imperfect":
            if self.sample_variant == "sample":
                self.q[np.arange(self.n_runs), self.s0_sampled, self.a1] += self.alpha * (
                    reward - self.q[np.arange(self.n_runs), self.s0_sampled, self.a1]
                )
            elif self.sample_variant == "expectation":
                for s in range(self.n_state):
                    self.q[np.arange(self.n_runs), s, self.a1] += (
                        self.alpha
                        * (reward - self.q[np.arange(self.n_runs), s, self.a1])
                        * self.p_s0_possible[:, s]
                    )
            else:
                raise ValueError("Incorrect sample variant")
        else:
            raise ValueError("Incorrect train variant")
        self.q0[np.arange(self.n_runs), self.s0, self.m0] += self.alpha * (
            reward - self.q0[np.arange(self.n_runs), self.s0, self.m0]
        )
        if self.sample_variant == "sample":
            self.q0_expected[np.arange(self.n_runs), self.s0_sampled, self.m0] += (
                self.alpha
                * (
                    reward
                    - self.q0_expected[np.arange(self.n_runs), self.s0_sampled, self.m0]
                )
            )
        elif self.sample_variant == "expectation":
            for s in range(self.n_state):
                self.q0_expected[np.arange(self.n_runs), s, self.m0] += (
                    self.alpha
                    * (reward - self.q0_expected[np.arange(self.n_runs), s, self.m0])
                    * self.p_s0_possible[:, s]
                )
        else:
            raise ValueError("Incorrect sample variant")

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
