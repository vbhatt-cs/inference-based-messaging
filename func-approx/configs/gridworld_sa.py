from ray.rllib.models import ModelCatalog

from ..agents.impala_cpc_sa import ImpalaCPCSaTrainer
from ..envs.treasure_hunt import TreasureHunt
from ..models.gridworld_sa_model import GridworldSaModel

ModelCatalog.register_custom_model("gridworld_sa_model", GridworldSaModel)

# Needed since dict obs is flattened
try:
    obs_shape = TreasureHunt().observation_space["obs"].shape
except TypeError:
    obs_shape = TreasureHunt().observation_space.shape

configs = dict()

config_impala = {
    # "run": "IMPALA",
    "run": ImpalaCPCSaTrainer,
    "tot_steps": 1e9,
    "max_workers": 4,
    "num_envs_per_worker": 8,
    "model": {
        "custom_model": "gridworld_sa_model",
        "conv_filters": [[6, 1, 1]],
        "conv_activation": "relu",
        "fcnet_activation": "relu",
        "fcnet_hiddens": [64, 64],
        "use_lstm": True,
        "lstm_cell_size": 128,
        "lstm_use_prev_action_reward": True,
        "max_seq_len": 100,
        "custom_options": {
            "obs_shape": obs_shape,
            "use_cpc": True,
            "cpc_opts": {"cpc_len": 20, "cpc_coeff": 10.0, "cpc_code_size": 64},
        },
    },
    "gamma": 0.99,
    "horizon": 500,
    "env_config": {
        "height": 18,
        "width": 24,
        "view_size": 5,
        "num_tunnels": 4,
        "goal_obs": True,
        "random_reset": True,
    },
    "env": TreasureHunt,
    "clip_actions": False,
    "log_level": "INFO",
    "eager": False,
    "evaluation_interval": 100,
    "evaluation_num_episodes": 10,
    "evaluation_config": {"explore": False},
    "sample_async": True,
    "sample_batch_size": 100,
    "train_batch_size": 16 * 100,
    "grad_clip": 10.0,
    "opt_type": "rmsprop",
    "lr": 1e-3,
    "lr_schedule": [
        (t, (0.99 ** int(t / 1e6)) * 1e-3) for t in range(0, int(1e9) + 1, int(1e6))
    ],  # Piecewise linear approx of exponential
    "epsilon": 1e-6,
    "entropy_coeff": 0.006,
    # "entropy_coeff_schedule": [
    #     (t, (0.99 ** int(t / 1e6)) * 0.006) for t in range(0, int(1e9) + 1, int(1e8))
    # ],  # Piecewise linear approx of exponential
    "vf_loss_coeff": 0.5,
}

configs["impala"] = config_impala
