import copy

from ray.rllib.models import ModelCatalog

from ..agents.impala_cpc_ma_comm import (
    SenderPolicy,
    MessageActionDistribution,
    ReceiverPolicy,
    DeterministicMessageActionDistribution,
)
from ..agents.impala_cpc_sa import ImpalaCPCSaTrainer
from ..envs.treasure_hunt import MultiTreasureHunt
from ..models.gridworld_ma_model import SenderModel, ReceiverModel

env_config = {
    "height": 18,
    "width": 24,
    "view_size": 5,
    "num_tunnels": 4,
    "message_size": 5,
    "random_reset": True,
}
sample_env = MultiTreasureHunt(env_config)

ModelCatalog.register_custom_model("sender_model", SenderModel)
ModelCatalog.register_custom_model("receiver_model", ReceiverModel)


# Hack for getting a custom multi action distribution
class SenderDist(MessageActionDistribution):
    def __init__(self, inputs, model):
        super().__init__(inputs, model, sample_env.action_space["sender"], "sender")


class ReceiverDist(MessageActionDistribution):
    def __init__(self, inputs, model):
        super().__init__(inputs, model, sample_env.action_space["receiver"], "receiver")


class DeterministicSenderDist(DeterministicMessageActionDistribution):
    def __init__(self, inputs, model):
        super().__init__(inputs, model, sample_env.action_space["sender"], "sender")


class DeterministicReceiverDist(DeterministicMessageActionDistribution):
    def __init__(self, inputs, model):
        super().__init__(inputs, model, sample_env.action_space["receiver"], "receiver")


ModelCatalog.register_custom_action_dist("sender_dist", SenderDist)
ModelCatalog.register_custom_action_dist("receiver_dist", ReceiverDist)
ModelCatalog.register_custom_action_dist(
    "deterministic_sender_dist", DeterministicSenderDist
)
ModelCatalog.register_custom_action_dist(
    "deterministic_receiver_dist", DeterministicReceiverDist
)

configs = []

default_config = {
    # "run": "IMPALA",
    "run": ImpalaCPCSaTrainer,
    "tot_steps": 1e9,
    "max_workers": 4,
    "num_envs_per_worker": 8,
    "model": {
        "conv_filters": [[6, 1, 1]],
        "conv_activation": "relu",
        "fcnet_activation": "relu",
        "fcnet_hiddens": [64, 64],
        "use_lstm": True,
        "lstm_cell_size": 128,
        "lstm_use_prev_action_reward": True,
        "max_seq_len": 100,
        "custom_options": {
            "obs_shape": sample_env.observation_space["sender"]["obs"].shape,
            "use_cpc": True,
            "cpc_opts": {"cpc_len": 20, "cpc_coeff": 10.0, "cpc_code_size": 64},
            "message_entropy_coeff": 0.0,
            "use_comm": True,
            "use_sender_bias": False,
            "sender_bias_opts": {
                "l_ps_lambda": 3.0,
                "entropy_target": 0.8,
                "sender_bias_coeff": 0.001,
            },
            "use_receiver_bias": False,
            "receiver_bias_opts": {"l_ce_coeff": 0.01, "l_pl_coeff": 0.003},
            "use_inference_policy": True,
            "inference_policy_opts": {
                # Type can be "moving_avg" or "hyper_nn"
                # - "moving_avg" calculates empirical moving average of unscaled p(m)
                #   "ewma_momentum" is the exponential weighting. None implies unbiased
                #   moving average
                # - "hyper_nn" uses a network with weights of message network as inputs
                #   to predict p(m)
                #   "pm_hidden is the list of hidden units in the MLP
                "type": "moving_avg",
                "ewma_momentum": 0.5,
                "pm_hidden": [64, 64],
            },
        },
    },
    "gamma": 0.99,
    "horizon": 500,
    "env_config": env_config,
    "env": MultiTreasureHunt,
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
    "multiagent": {
        # Map from policy ids to tuples of (policy_cls, obs_space,
        # act_space, config). See rollout_worker.py for more info.
        "policies": {
            "sender": (
                SenderPolicy,
                sample_env.observation_space["sender"],
                sample_env.action_space["sender"],
                {
                    "model": {
                        "custom_model": "sender_model",
                        "custom_action_dist": "sender_dist",
                    }
                },
            ),
            "receiver": (
                ReceiverPolicy,
                sample_env.observation_space["receiver"],
                sample_env.action_space["receiver"],
                {
                    "model": {
                        "custom_model": "receiver_model",
                        "custom_action_dist": "receiver_dist",
                    }
                },
            ),
        },
        # Function mapping agent ids to policy ids.
        "policy_mapping_fn": lambda x: x,
    },
}

configs.append(default_config)

# config1 = copy.deepcopy(default_config)
# config1["model"]["custom_options"]["use_sender_bias"] = True
# configs.append(config1)
#
# config2 = copy.deepcopy(default_config)
# config2["model"]["custom_options"]["use_receiver_bias"] = True
# configs.append(config2)
#
# config3 = copy.deepcopy(default_config)
# config3["model"]["custom_options"]["use_sender_bias"] = True
# config3["model"]["custom_options"]["use_receiver_bias"] = True
# configs.append(config3)

config1 = copy.deepcopy(default_config)
config1["multiagent"]["policies"] = {
    "sender": (
        SenderPolicy,
        sample_env.observation_space["sender"],
        sample_env.action_space["sender"],
        {
            "model": {
                "custom_model": "sender_model",
                "custom_action_dist": "deterministic_sender_dist",
            }
        },
    ),
    "receiver": (
        ReceiverPolicy,
        sample_env.observation_space["receiver"],
        sample_env.action_space["receiver"],
        {
            "model": {
                "custom_model": "receiver_model",
                "custom_action_dist": "deterministic_receiver_dist",
            }
        },
    ),
}
configs.append(config1)

config2 = copy.deepcopy(default_config)
config2["model"]["custom_options"]["inference_policy_opts"]["ewma_momentum"] = 0.75
configs.append(config2)

config3 = copy.deepcopy(config1)
config3["model"]["custom_options"]["inference_policy_opts"]["ewma_momentum"] = 0.75
configs.append(config3)
