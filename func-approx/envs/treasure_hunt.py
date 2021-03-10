import warnings
from enum import IntEnum

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Discrete, Box, Dict, Tuple
from ray.rllib import MultiAgentEnv


class TreasureHunt(gym.Env):
    """
    Single agent version of the gridworld environment as specified in Eccles, Tom, et al.
    "Biases for Emergent Communication in Multi-agent Reinforcement Learning." Advances
    in Neural Information Processing Systems. 2019.
    """

    class StateTypes(IntEnum):
        wall = 0
        open = 1
        goal = 2
        agent1 = 3

    class ActionTypes(IntEnum):
        up = 0
        down = 1
        left = 2
        right = 3
        noop = 4

    def __init__(self, env_config=None):
        """
        Args:
            env_config: Dict with following keys (all are optional):
                width: Width of the grid (default 24)
                height: Height of the grid (default 18)
                view_size: Size of observations (default 5)
                num_tunnels: Number of tunnels (default 4)
                random_reset: True if the grid is fully re-generated after every
                    episode (default True)
                horizon: Length of an episode (default 500)
                goal_obs: True if goal tunnel should be in observation (default False)
                seed: Random seed to use (default None)
        """
        self.full_env_config = dict(
            width=24,
            height=18,
            view_size=5,
            num_tunnels=4,
            random_reset=True,
            horizon=500,
            goal_obs=False,
            seed=None,
        )
        if env_config is not None:
            for k, v in env_config.items():
                if k not in self.full_env_config:
                    warnings.warn("Argument in env_config unknown")
                self.full_env_config[k] = v

        self.width = self.full_env_config["width"]
        self.height = self.full_env_config["height"]
        self.view_size = self.full_env_config["view_size"]
        self.num_tunnels = self.full_env_config["num_tunnels"]
        self.random_reset = self.full_env_config["random_reset"]
        self.horizon = int(self.full_env_config["horizon"])
        self.goal_obs = self.full_env_config["goal_obs"]
        self.rng = np.random.RandomState(self.full_env_config["seed"])
        assert self.view_size % 2 != 0

        obs_shape = (self.view_size, self.view_size, len(self.StateTypes))
        if self.goal_obs:
            self.observation_space = Dict(
                {
                    "obs": Box(low=0, high=1, shape=obs_shape, dtype=np.int),
                    "goal": Box(low=0, high=1, shape=(4,), dtype=np.int),
                }
            )
        else:
            self.observation_space = Box(low=0, high=1, shape=obs_shape, dtype=np.int)

        self.action_space = Discrete(len(self.ActionTypes))

        self.state = None  # Position of the agent
        self.goal_state = None  # Terminal state
        self.t = 0

        self.init_grid = None  # For saving the map
        self.grid = None  # Current map
        self.pad_size = self.view_size // 2  # Padding to the grid for efficient obs
        self.padded_grid = None  # For calculating obs
        self.goal_hidden = False  # Hack for when goal spawns under the agent

        self.ax = None
        self.colors = np.array(
            [
                (0.5, 0.5, 0.5),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 0.0),
            ]
        )

    def reset(self):
        """
        Reset the environment and return the starting observations
        Returns:
            Starting observation
        """
        if self.init_grid is None or self.random_reset:  # Re-generate map
            self.grid = np.zeros((self.width, self.height), dtype=np.int)
            self._generate_tunnels()

            # Agent
            agent_x = self.rng.randint(1, self.width - 1)
            self.state = (agent_x, 1)
            self.grid[self.state] = self.StateTypes.agent1
            self.init_grid = self.grid.copy()
        else:  # Copy the map from before
            self.grid = self.init_grid.copy()
            x1, y1 = np.where(self.grid == self.StateTypes.agent1)
            self.state = (x1[0], y1[0])
            self.goal_state = list(zip(*np.where(self.grid == self.StateTypes.goal)))[0]

        self.t = 0

        if self.goal_obs:
            obs = {
                "obs": self._get_obs(self.state, self.view_size),
                "goal": self._get_goal(),
            }
        else:
            obs = self._get_obs(self.state, self.view_size)
        return obs

    def step(self, action):
        """
        Advance the environment by one step
        Args:
            action: Action of the agent

        Returns:
            next state, reward, if the episode is done, None
        """
        x1, y1, waste = self._move_agent(self.state, action)
        self.grid[self.state] = self.StateTypes.open
        self.grid[x1, y1] = self.StateTypes.agent1

        reward = self._handle_goal_step(x1, y1)
        self.state = (x1, y1)
        done = False
        self.t += 1

        if self.t == self.horizon:
            done = True

        if self.goal_obs:
            obs = {
                "obs": self._get_obs(self.state, self.view_size),
                "goal": self._get_goal(),
            }
        else:
            obs = self._get_obs(self.state, self.view_size)
        return obs, reward, done, {"waste": waste}

    def render(self, mode="human"):
        """
        Render the environment for visualization
        """
        colored_grid = self.colors[self.grid.T]
        if mode == "rgb_array":
            return colored_grid
        elif mode == "human":
            if self.ax is None:
                self._init_rendering()
            self.ax.imshow(colored_grid)
            plt.pause(0.1)
            plt.draw()
        else:
            super().render(mode=mode)  # just raise an exception

    def _init_rendering(self):
        """
        Initialize window for rendering
        """
        fig, ax = plt.subplots()
        ax.set_axis_off()
        self.ax = ax

    def _generate_tunnels(self):
        """
        Generate tunnels and goal
        Returns:
            None (self.grid is modified to have the tunnels and goal,
                self.goal_state has the current goal state)
        """
        self.grid[1:-1, 1] = self.StateTypes.open  # Top tunnel
        self.grid[1:-1, -2] = self.StateTypes.open  # Bottom tunnel

        # Vertical tunnels should be at least 3 pixels apart
        d = 3
        while True:
            tunnel_xs = np.sort(
                self.rng.randint(1, self.width - 1, size=self.num_tunnels)
            )
            ds = tunnel_xs[1:] - tunnel_xs[:-1]
            if np.all(ds >= d):
                break

        goal_tunnel = self.rng.randint(self.num_tunnels)
        for i in range(self.num_tunnels):
            self.grid[tunnel_xs[i], 1:-3] = self.StateTypes.open
            if goal_tunnel == i:
                self.grid[tunnel_xs[i], -4] = self.StateTypes.goal
                self.goal_state = (tunnel_xs[i], self.height - 4)

    def _get_obs(self, center, view_size):
        """
        Calculate the observations
        Args:
            center: Position of the agent
            view_size: Size of the observation square

        Returns:
            Matrix of objects around the agent
        """
        if self.padded_grid is None:
            self.padded_grid = (
                np.zeros(
                    (self.width + 2 * self.pad_size, self.height + 2 * self.pad_size),
                    dtype=np.int,
                )
                + self.StateTypes.wall
            )

        self.padded_grid[
            self.pad_size : self.pad_size + self.width,
            self.pad_size : self.pad_size + self.height,
        ] = self.grid

        top = center[1]
        bottom = center[1] + view_size
        left = center[0]
        right = center[0] + view_size
        obs = self.padded_grid[left:right, top:bottom]
        onehot_obs = (np.arange(len(self.StateTypes)) == obs[..., None]).astype(np.int)
        return onehot_obs

    def _get_goal(self):
        tunnel_xs = np.hstack(
            [np.where(self.grid[:, -4] == self.StateTypes.open)[0], self.goal_state[0]]
        )
        goal_tunnel = np.where(np.sort(tunnel_xs) == self.goal_state[0])
        # goal_tunnel = np.random.randint(4)
        goal_tunnel_onehot = np.zeros(4, dtype=np.int)
        goal_tunnel_onehot[goal_tunnel] = 1
        return goal_tunnel_onehot

    def _move_agent(self, center, action):
        """
        Move an agent given the action
        Args:
            center: Position of the agent
            action: Action taken

        Returns:
            New position after the action
        """
        x, y = center

        if action == self.ActionTypes.up:
            y = max(0, y - 1)
        elif action == self.ActionTypes.down:
            y = min(self.height - 1, y + 1)
        elif action == self.ActionTypes.right:
            x = min(self.width - 1, x + 1)
        elif action == self.ActionTypes.left:
            x = max(0, x - 1)

        if self.grid[x, y] in [self.StateTypes.open, self.StateTypes.goal]:
            return x, y, False
        else:
            return (*center, True)

    def _handle_goal_step(self, x1, y1):
        if self.goal_hidden and self.grid[self.goal_state] == self.StateTypes.open:
            self.grid[self.goal_state] = self.StateTypes.goal
            self.goal_hidden = False
        if self.goal_state == (x1, y1):
            reward = 1
            new_goal_x = self.rng.choice(
                np.hstack(
                    [
                        np.where(self.grid[:, -4] == self.StateTypes.open)[0],
                        self.goal_state[0],
                    ]
                )
            )
            self.goal_state = (new_goal_x, self.height - 4)
            if self.grid[self.goal_state] == self.StateTypes.open:
                self.grid[self.goal_state] = self.StateTypes.goal
                self.goal_hidden = False
            else:
                self.goal_hidden = True
        else:
            reward = 0
        return reward


class MultiTreasureHunt(MultiAgentEnv, TreasureHunt):
    """
    Gridworld environment as specified in Eccles, Tom, et al.
    "Biases for Emergent Communication in Multi-agent Reinforcement Learning." Advances
    in Neural Information Processing Systems. 2019.
    """

    class StateTypes(IntEnum):
        wall = 0
        open = 1
        goal = 2
        agent1 = 3
        agent2 = 4

    def __init__(self, env_config=None):
        """
        Args:
            env_config: Dict with following keys (all are optional):
                width: Width of the grid (default 24)
                height: Height of the grid (default 18)
                view_size: Size of observations (default 5)
                num_tunnels: Number of tunnels (default 4)
                random_reset: True if the grid is fully re-generated after every
                    episode (default True)
                horizon: Length of an episode (default 500)
                seed: Random seed to use (default None)
        """
        super().__init__(env_config)
        self.agent1 = "receiver"
        self.agent2 = "sender"
        self.message_size = self.full_env_config.get("message_size", 5)
        obs_shape = (self.view_size, self.view_size, len(self.StateTypes))
        self.observation_space = Dict(
            {
                self.agent1: Dict(
                    {
                        "obs": Box(low=0, high=1, shape=obs_shape, dtype=np.int),
                        "message": Box(
                            low=0, high=1, shape=(self.message_size,), dtype=np.int
                        ),
                    }
                ),
                self.agent2: Dict(
                    {
                        "obs": Box(low=0, high=1, shape=obs_shape, dtype=np.int),
                        "message": Box(
                            low=0, high=1, shape=(self.message_size,), dtype=np.int
                        ),
                    }
                ),
            }
        )
        self.action_space = Dict(
            {
                self.agent1: Tuple(
                    (Discrete(len(self.ActionTypes)), Discrete(self.message_size))
                ),
                self.agent2: Tuple(
                    (Discrete(len(self.ActionTypes)), Discrete(self.message_size))
                ),
            }
        )

    def reset(self):
        """
        Reset the environment and return the starting observations
        Returns:
            Starting observations
        """
        if self.init_grid is None or self.random_reset:  # Re-generate map
            self.grid = np.zeros((self.width, self.height), dtype=np.int)
            self._generate_tunnels()

            # Agents
            agent_x = self.rng.randint(1, self.width - 1, size=2)
            self.state = [(agent_x[0], 1), (agent_x[1], self.height - 2)]
            self.grid[self.state[0]] = self.StateTypes.agent1
            self.grid[self.state[1]] = self.StateTypes.agent2
            self.init_grid = self.grid.copy()
        else:  # Copy the map from before
            self.grid = self.init_grid.copy()
            x1, y1 = np.where(self.grid == self.StateTypes.agent1)
            x2, y2 = np.where(self.grid == self.StateTypes.agent2)
            self.state = [(x1[0], y1[0]), (x2[0], y2[0])]
            self.goal_state = list(zip(*np.where(self.grid == self.StateTypes.goal)))[0]

        self.t = 0
        obs = {
            self.agent1: {
                "obs": self._get_obs(self.state[0], self.view_size),
                "message": np.zeros(self.message_size, dtype=np.int),
            },
            self.agent2: {
                "obs": self._get_obs(self.state[1], self.view_size),
                "message": np.zeros(self.message_size, dtype=np.int),
            },
        }
        return obs

    def step(self, action_dict):
        """
        Advance the environment by one step
        Args:
            action_dict: Dict of actions of both the agents

        Returns:
            next state, reward, if the episode is done, None
        """
        action1 = action_dict[self.agent1][0]
        message1 = np.zeros(self.message_size, dtype=np.int)
        message1[action_dict[self.agent1][1]] = 1
        action2 = action_dict[self.agent2][0]
        message2 = np.zeros(self.message_size, dtype=np.int)
        message2[action_dict[self.agent2][1]] = 1
        x1, y1, waste = self._move_agent(self.state[0], action1)
        self.grid[self.state[0]] = self.StateTypes.open
        self.grid[x1, y1] = self.StateTypes.agent1
        x2, y2, _ = self._move_agent(self.state[1], action2)
        self.grid[self.state[1]] = self.StateTypes.open
        self.grid[x2, y2] = self.StateTypes.agent2

        reward = self._handle_goal_step(x1, y1)
        self.state = [(x1, y1), (x2, y2)]
        done = False
        self.t += 1

        if self.t == self.horizon:
            done = True

        obs = {
            self.agent1: {
                "obs": self._get_obs(self.state[0], self.view_size),
                "message": message2,
            },
            self.agent2: {
                "obs": self._get_obs(self.state[1], self.view_size),
                "message": message1,
            },
        }
        rewards = {
            self.agent1: reward,
            self.agent2: reward,
        }
        dones = {
            self.agent1: done,
            self.agent2: done,
            "__all__": done,
        }
        infos = {
            self.agent1: {"waste": waste},
            self.agent2: {"waste": waste},
        }
        return obs, rewards, dones, infos
