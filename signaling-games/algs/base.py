import numpy as np


class BaseAlg:
    """
    Base algorithm. All algorithms need to define act and train methods.
    """

    def act(self, state, test=False):
        raise NotImplementedError

    def train(self, reward):
        raise NotImplementedError

    @staticmethod
    def generate_configs(default_config):
        """
        An optional method that generates the list of configs for hyperparameter tuning
        Args:
            default_config (dict): Default values of the parameters
        Returns:
            List of dicts with the required parameters
        """
        return [default_config]


class RandomPolicy(BaseAlg):
    """
    Random policy
    """

    def __init__(self, n_state, n_mess, n_act, n_runs, **kwargs):
        """
        Args:
            n_state (int): Number of states
            n_mess (int): Number of messages
            n_act (int): Number of actions
            n_runs (int): Number of runs
        """
        self.n_state = n_state
        self.n_mess = n_mess
        self.n_act = n_act
        self.n_runs = n_runs

    def act(self, state, test=False):
        m0 = np.random.randint(self.n_mess, size=self.n_runs)
        a1 = np.random.randint(self.n_act, size=self.n_runs)

        return m0, a1

    def train(self, reward):
        return
