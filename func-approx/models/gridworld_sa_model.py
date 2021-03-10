import numpy as np
from ray.rllib.models.tf.misc import get_activation_fn
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.visionnet_v1 import _get_filter_config
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils import try_import_tf

from ..utils.model_utils import build_cnn, build_fc, build_lstm

tf = try_import_tf()


class GridworldSaModel(TFModelV2):
    """
    Single agent recurrent model for Gridworld env
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        with tf.variable_scope("sa_model", reuse=tf.AUTO_REUSE):
            super().__init__(obs_space, action_space, num_outputs, model_config, name)
            custom_opts = model_config.get("custom_options", {})

            obs_space_shape = custom_opts.get("obs_shape", obs_space.shape)
            if len(obs_space.shape) == 1:
                n_extra_obs = obs_space.shape[0] - np.prod(obs_space_shape)
            else:
                n_extra_obs = 0

            # Conv
            activation = get_activation_fn(model_config.get("conv_activation"))
            filters = model_config.get("conv_filters")
            if filters is None:
                filters = _get_filter_config(obs_space_shape)

            inputs = tf.keras.layers.Input(
                shape=(None, *obs_space_shape), name="observations_time"
            )
            model_inputs = [inputs]

            cnn_in = tf.reshape(inputs, [-1, *obs_space_shape])
            conv_out = build_cnn(cnn_in, filters, activation, name="conv")

            # FC
            activation = get_activation_fn(model_config.get("fcnet_activation"))
            hiddens = model_config.get("fcnet_hiddens")

            if n_extra_obs > 0:
                extra_inputs = tf.keras.layers.Input(
                    shape=(n_extra_obs,), name="extra_observations"
                )
                model_inputs.append(extra_inputs)
                fc_in = tf.keras.layers.Concatenate(name="fc_in")(
                    [tf.keras.layers.Flatten()(conv_out), extra_inputs]
                )
            else:
                fc_in = tf.keras.layers.Flatten(name="fc_in")(conv_out)
            fc_out = build_fc(fc_in, hiddens, activation, name="fc")

            # LSTM
            self.cell_size = model_config.get("lstm_cell_size", 256)

            state_in_h = tf.keras.layers.Input(shape=(self.cell_size,), name="h")
            state_in_c = tf.keras.layers.Input(shape=(self.cell_size,), name="c")
            seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

            prev_actions = tf.keras.layers.Input(
                shape=(), name="prev_actions", dtype=tf.int32
            )
            prev_rewards = tf.keras.layers.Input(shape=(), name="prev_rewards")

            model_inputs.extend(
                [prev_actions, prev_rewards, seq_in, state_in_h, state_in_c]
            )

            if model_config.get("lstm_use_prev_action_reward"):
                prev_actions_onehot = tf.one_hot(prev_actions, action_space.n)
                in_tensors = [fc_out, prev_actions_onehot, prev_rewards]
            else:
                in_tensors = [fc_out]

            # CPC objective
            self.use_cpc = custom_opts.get("use_cpc", False)
            cpc_len = custom_opts.get("cpc_len", 20)
            cpc_code_size = custom_opts.get("cpc_code_size", 64)
            self.cpc_in_shape = [cpc_code_size]
            self.cpc_out_shape = [cpc_len, cpc_code_size]
            cpc_params = {
                "cpc_len": cpc_len,
                "name": "cpc",
                "cpc_code_size": cpc_code_size,
            }

            lstm_out, model_outputs = build_lstm(
                in_tensors,
                state_in_h=state_in_h,
                state_in_c=state_in_c,
                seq_in=seq_in,
                cell_size=self.cell_size,
                add_cpc=self.use_cpc,
                cpc_params=cpc_params,
                name="lstm",
            )
            # The actual CPC encodings
            self._cpc_ins = None
            self._cpc_preds = None

            # Final layer
            logits = tf.keras.layers.Dense(
                num_outputs, activation=tf.keras.activations.linear, name="logits"
            )(lstm_out)
            values = tf.keras.layers.Dense(1, activation=None, name="values")(lstm_out)
            self._value_out = None  # The actual value
            model_outputs = [logits, values] + model_outputs

            # Create the RNN model
            self.rnn_model = tf.keras.Model(inputs=model_inputs, outputs=model_outputs)
            self.register_variables(self.rnn_model.variables)
            self._model_out = None  # Actual logits
            self.rnn_model.summary()

    def forward(self, input_dict, state, seq_lens):
        """
        Adds time dimension to batch and does forward inference
        """
        prev_actions = tf.cast(input_dict["prev_actions"], dtype=tf.int32)
        prev_rewards = input_dict["prev_rewards"]

        if isinstance(input_dict["obs"], dict):
            obs_dict = input_dict["obs"]
            inputs = add_time_dimension(obs_dict["obs"], seq_lens)
            extra_inputs = obs_dict["goal"]
            outputs = self.rnn_model(
                [inputs, extra_inputs, prev_actions, prev_rewards, seq_lens] + state
            )
        else:
            inputs = add_time_dimension(input_dict["obs"], seq_lens)
            outputs = self.rnn_model(
                [inputs, prev_actions, prev_rewards, seq_lens] + state
            )

        if self.use_cpc:
            model_out, self._value_out, h, c, self._cpc_ins, self._cpc_preds = outputs
        else:
            model_out, self._value_out, h, c = outputs

        self._model_out = tf.reshape(model_out, [-1, self.num_outputs])
        return self._model_out, [h, c]

    def get_initial_state(self):
        """See parent class for docs"""
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    def value_function(self):
        """See parent class for docs"""
        assert (
            self._value_out is not None
        ), "Call forward() before calling value_function()"
        return tf.reshape(self._value_out, [-1])

    def cpc_ins(self):
        """Returns the CPC input encodings"""
        if self.use_cpc:
            return tf.reshape(self._cpc_ins, [-1] + self.cpc_in_shape)
        else:
            raise ValueError("Model doesn't use CPC")

    def cpc_preds(self):
        """Returns the CPC output encodings at next k time steps"""
        if self.use_cpc:
            return tf.reshape(self._cpc_preds, [-1] + self.cpc_out_shape)
        else:
            raise ValueError("Model doesn't use CPC")
