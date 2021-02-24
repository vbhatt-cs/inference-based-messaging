import numpy as np

from ... import utils
from ..base import BaseAlg


class CommBias(BaseAlg):
    """
    Training using biases for communication.
    Eccles, Tom, et al. "Biases for emergent communication in multi-agent reinforcement
    learning." arXiv preprint arXiv:1912.05676 (2019).
    """

    def __init__(
            self,
            n_state,
            n_mess,
            n_act,
            n_runs,
            alpha0=0.1,
            alpha_critic0=0.1,
            bias_type="sender",
            l_ps_coeff=0.001,
            l_ps_lambda=3.0,
            entropy_target=0.8,
            l_pl_coeff=0.1,
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
            l_ps_coeff (float): Weight given to sender bias
            l_ps_lambda (float): Weight given to average message entropy
            entropy_target (float): Target entropy
            mode (int): 0 for communication, 1 for fixed messages, 2 for fixed actions
        """
        self.n_state = n_state
        self.n_mess = n_mess
        self.n_act = n_act
        self.n_runs = n_runs
        self.alpha = alpha0
        self.alpha_critic = alpha_critic0
        self.bias_type = bias_type
        self.l_ps_coeff = l_ps_coeff
        self.l_ps_lambda = l_ps_lambda
        self.entropy_target = entropy_target
        self.l_pl_coeff = l_pl_coeff
        self.mode = mode

        self.h0 = np.random.normal(scale=1 / self.n_state, size=(n_runs, n_state, n_mess))
        self.v0 = np.zeros((n_runs, n_state))
        self.h1 = np.random.normal(scale=1 / self.n_mess, size=(n_runs, n_mess, n_act))
        self.v1 = np.zeros((n_runs, n_mess))

        self.s0 = None
        self.s1 = None
        self.m0 = None
        self.a1 = None
        self.pi0_s = None
        self.pi1_s = None

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
            self.pi0_s = np.zeros((self.n_runs, self.n_mess)) + 1e-10
            self.pi0_s[np.arange(self.n_runs), self.m0] = 1 - 1e-10 * (self.n_mess - 1)
        else:
            h0_s = self.h0[np.arange(self.n_runs), state]
            self.pi0_s = utils.softmax(h0_s, axis=1)
            self.m0 = utils.vectorized_2d_choice(np.arange(self.n_mess), p=self.pi0_s)

        self.s1 = self.m0

        if self.mode == 2:
            self.a1 = self.s1
            self.pi1_s = np.zeros((self.n_runs, self.n_act)) + 1e-10
            self.pi1_s[np.arange(self.n_runs), self.a1] = 1 - 1e-10 * (self.n_act - 1)
        else:
            h1_s = self.h1[np.arange(self.n_runs), self.s1]
            self.pi1_s = utils.softmax(h1_s, axis=1)
            self.a1 = utils.vectorized_2d_choice(np.arange(self.n_act), p=self.pi1_s)

        return self.m0, self.a1

    def train(self, reward):
        """
        Update the policy
        Args:
            reward (np.ndarray): Reward obtained (size=n_runs)
        """
        delta0 = reward - self.v0[np.arange(self.n_runs), self.s0]
        # delta0 = reward
        self.v0[np.arange(self.n_runs), self.s0] += self.alpha_critic * delta0
        policy_grad = -self.pi0_s
        policy_grad[np.arange(self.n_runs), self.m0] += 1

        if self.bias_type in ["sender", "both"]:
            pi0 = utils.softmax(self.h0, axis=-1)

            avg_message_grad = -np.einsum("ijk,ijl->ijkl", pi0, pi0)
            for i in range(self.n_mess):
                avg_message_grad[:, :, i, i] += pi0[:, :, i]

            avg_message_entropy_grad = -np.einsum(
                "ijl,ijkl->ijk", 1 + np.log(pi0), avg_message_grad
            )

            message_grad = -np.einsum("ij,ik->ijk", self.pi0_s, self.pi0_s)
            for i in range(self.n_mess):
                message_grad[:, i, i] += self.pi0_s[:, i]

            entropy_grad = -np.einsum("ik,ijk->ij", 1 + np.log(self.pi0_s), message_grad)
            entropy = -np.sum(self.pi0_s * np.log(self.pi0_s), axis=-1, keepdims=True)
            target_entropy_grad = 2 * (entropy - self.entropy_target) * entropy_grad

            total_grad = np.zeros((self.n_runs, self.n_state, self.n_mess))
            total_grad[:] = self.l_ps_coeff * self.l_ps_lambda * avg_message_entropy_grad
            total_grad[np.arange(self.n_runs), self.s0] += (
                    delta0[:, np.newaxis] * policy_grad
                    - self.l_ps_coeff * target_entropy_grad
            )
        else:
            total_grad = np.zeros((self.n_runs, self.n_state, self.n_mess))
            total_grad[np.arange(self.n_runs), self.s0] += (
                    delta0[:, np.newaxis] * policy_grad
            )

        self.h0 += self.alpha * total_grad

        delta1 = reward - self.v1[np.arange(self.n_runs), self.s1]
        # delta1 = reward
        self.v1[np.arange(self.n_runs), self.s1] += self.alpha_critic * delta1
        grad = -self.pi1_s
        grad[np.arange(self.n_runs), self.a1] += 1

        if self.bias_type in ["receiver", "both"]:
            pi1_bar_s = np.ones((self.n_runs, self.n_act)) / self.n_act
            action_grad = -np.einsum("ij,ik->ijk", self.pi1_s, self.pi1_s)
            for i in range(self.n_mess):
                action_grad[:, i, i] += self.pi1_s[:, i]

            l_pl_grad = np.sum(
                np.sign(self.pi1_s - pi1_bar_s)[:, :, np.newaxis] * action_grad, axis=-1
            )
            total_grad = delta1[:, np.newaxis] * grad + self.l_pl_coeff * l_pl_grad
        else:
            total_grad = delta1[:, np.newaxis] * grad

        self.h1[np.arange(self.n_runs), self.s1] += self.alpha * total_grad

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
        for l_ps_coeff in [1e-3, 0.01, 0.1]:
            for l_ps_lambda in [0.1, 0.3, 1.0]:
                for entropy_target in [0.0, 0.5, 1.0, 1.5]:
                    for l_pl_coeff in [1e-4, 0.001, 0.01, 0.1]:
                        for alpha in [0.1, 0.5]:
                            config = default_config.copy()
                            config["alpha0"] = alpha
                            config["alpha_critic0"] = alpha
                            config["l_ps_coeff"] = l_ps_coeff
                            config["l_ps_lambda"] = l_ps_lambda
                            config["entropy_target"] = entropy_target
                            config["l_pl_coeff"] = l_pl_coeff
                            configs.append(config)
        return configs
