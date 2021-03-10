import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.misc import get_activation_fn
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.visionnet_v1 import _get_filter_config
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils import try_import_tf

from ..utils.model_utils import build_cnn, build_fc, build_lstm

tf = try_import_tf()


class GridworldMaModel(TFModelV2):
    """
    Multi agent recurrent model for Gridworld env
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        with tf.variable_scope(f"{name}_model", reuse=tf.AUTO_REUSE):
            super().__init__(obs_space, action_space, num_outputs, model_config, name)
            custom_opts = model_config.get("custom_options", {})
            self.use_comm = custom_opts.get("use_comm", True)

            self.message_coeff = custom_opts.get("message_entropy_coeff", 0.0)
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
                shape=(None, *obs_space_shape), name=f"{name}_observations_time"
            )
            model_inputs = [inputs]

            cnn_in = tf.reshape(inputs, [-1, *obs_space_shape])
            conv_out = build_cnn(cnn_in, filters, activation, name=f"{name}_conv")

            # FC
            activation = get_activation_fn(model_config.get("fcnet_activation"))
            hiddens = model_config.get("fcnet_hiddens")

            if n_extra_obs > 0:
                extra_inputs = tf.keras.layers.Input(
                    shape=(n_extra_obs,), name=f"{name}_extra_observations"
                )
                model_inputs.append(extra_inputs)
                fc_in = tf.keras.layers.Concatenate(name=f"{name}_fc_in")(
                    [tf.keras.layers.Flatten()(conv_out), extra_inputs]
                )
            else:
                fc_in = tf.keras.layers.Flatten(name=f"{name}_fc_in")(conv_out)
            fc_out = build_fc(fc_in, hiddens, activation, name=f"{name}_fc")

            # LSTM
            self.cell_size = model_config.get("lstm_cell_size", 256)

            state_in_h = tf.keras.layers.Input(shape=(self.cell_size,), name=f"{name}_h")
            state_in_c = tf.keras.layers.Input(shape=(self.cell_size,), name=f"{name}_c")
            seq_in = tf.keras.layers.Input(
                shape=(), name=f"{name}_seq_in", dtype=tf.int32
            )

            prev_actions = tf.keras.layers.Input(
                shape=(), name=f"{name}_prev_actions", dtype=tf.int32
            )
            prev_rewards = tf.keras.layers.Input(shape=(), name=f"{name}_prev_rewards")

            model_inputs.extend(
                [prev_actions, prev_rewards, seq_in, state_in_h, state_in_c]
            )

            if model_config.get("lstm_use_prev_action_reward"):
                prev_actions_onehot = tf.one_hot(prev_actions, action_space[0].n)
                in_tensors = [fc_out, prev_actions_onehot, prev_rewards]
            else:
                in_tensors = [fc_out]

            # CPC objective
            self.use_cpc = custom_opts.get("use_cpc", False)
            if self.use_cpc:
                cpc_params = custom_opts["cpc_opts"]
                self.cpc_in_shape = [cpc_params["cpc_code_size"]]
                self.cpc_out_shape = [cpc_params["cpc_len"], cpc_params["cpc_code_size"]]
                cpc_params["name"] = f"{name}_cpc"
                # The actual CPC encodings
                self._cpc_ins = None
                self._cpc_preds = None
            else:
                cpc_params = {}

            lstm_out, model_outputs = build_lstm(
                in_tensors,
                state_in_h=state_in_h,
                state_in_c=state_in_c,
                seq_in=seq_in,
                cell_size=self.cell_size,
                add_cpc=self.use_cpc,
                cpc_params=cpc_params,
                name=f"{name}_lstm",
            )

            # Final layer, logits has both actions and messages
            self.use_inference_policy = custom_opts.get("use_inference_policy", False)
            if self.use_inference_policy:
                inference_policy_opts = custom_opts["inference_policy_opts"]
                self.pm_type = inference_policy_opts["type"]
                self.ewma_momentum = inference_policy_opts.get("ewma_momentum")
                self.pm_hidden = inference_policy_opts.get("pm_hidden", [64, 64])
                self.message_size = action_space[1].n

                action_logits = tf.keras.layers.Dense(
                    action_space[0].n,
                    activation=tf.keras.activations.linear,
                    name=f"{name}_action_logits",
                )(lstm_out)
                unscaled_message_logits = tf.keras.layers.Dense(
                    self.message_size,
                    activation=tf.keras.activations.linear,
                    name=f"{name}_unscaled_message_logits",
                )(lstm_out)
                unscaled_message_p = tf.nn.softmax(unscaled_message_logits)
                model_outputs.append(unscaled_message_p)

                if self.pm_type == "moving_avg":
                    self._avg_message_p = tf.Variable(
                        name=f"{name}_avg",
                        initial_value=tf.ones((self.message_size,)) / self.message_size,
                        trainable=False,
                    )
                    avg_message_vars = [self._avg_message_p]
                    if self.ewma_momentum is None:
                        self._avg_message_t = tf.Variable(
                            name=f"{name}_t", initial_value=tf.zeros(()), trainable=False,
                        )
                        avg_message_vars.append(self._avg_message_t)

                    self.register_variables(avg_message_vars)

                logits = tf.keras.layers.Concatenate(name=f"{name}_logits")(
                    [action_logits, unscaled_message_logits]
                )
            else:
                logits = tf.keras.layers.Dense(
                    num_outputs,
                    activation=tf.keras.activations.linear,
                    name=f"{name}_logits",
                )(lstm_out)

            values = tf.keras.layers.Dense(1, activation=None, name=f"{name}_values")(
                lstm_out
            )
            self._value_out = None  # The actual value
            model_outputs = [logits, values] + model_outputs

            # Create the RNN model
            self.rnn_model = tf.keras.Model(inputs=model_inputs, outputs=model_outputs)
            self.register_variables(self.rnn_model.variables)
            self._model_out = None  # Actual logits
            self.rnn_model.summary()

            if self.use_inference_policy and self.pm_type == "hyper_nn":
                flattened_vars = []
                message_model = tf.keras.Model(
                    inputs=model_inputs, outputs=unscaled_message_logits
                )
                for e in message_model.variables:
                    flattened_vars.append(tf.reshape(tf.stop_gradient(e), shape=[1, -1]))

                concat_vars = tf.keras.layers.Concatenate()(flattened_vars)
                pm_fc_out, pm_fc_vars = build_fc(
                    concat_vars, self.pm_hidden, "relu", name="pm_fc", return_vars=True
                )
                pm_logits_layer = tf.keras.layers.Dense(
                    self.message_size,
                    activation=tf.keras.activations.linear,
                    name=f"{name}_pm_logits",
                )
                self._pm_logits = pm_logits_layer(pm_fc_out)
                self.register_variables(pm_fc_vars)
                self.register_variables(pm_logits_layer.variables)

            # Extra variable definitions
            self.use_receiver_bias = custom_opts.get("use_receiver_bias", False)
            self.no_message_outputs = None
            self._unscaled_message_p = None

    def forward(self, input_dict, state, seq_lens):
        """
        Adds time dimension to batch and does forward inference
        """
        prev_actions = tf.cast(input_dict["prev_actions"][:, 0], dtype=tf.int32)
        prev_rewards = input_dict["prev_rewards"]
        lstm_state = state[:2]
        if self.use_receiver_bias:
            receiver_bias_state = state[2:4]

        obs_dict = input_dict["obs"]
        inputs = add_time_dimension(obs_dict["obs"], seq_lens)
        if self.use_comm:
            extra_inputs = obs_dict["message"]
        else:
            extra_inputs = tf.zeros_like(obs_dict["message"])
        outputs = self.rnn_model(
            [inputs, extra_inputs, prev_actions, prev_rewards, seq_lens] + lstm_state
        )

        if self.use_receiver_bias:
            extra_inputs = tf.zeros_like(obs_dict["message"])
            self.no_message_outputs = self.rnn_model(
                [inputs, extra_inputs, prev_actions, prev_rewards, seq_lens]
                + receiver_bias_state
            )

        if self.use_cpc:
            (
                model_out,
                self._value_out,
                h,
                c,
                self._cpc_ins,
                self._cpc_preds,
                *self._unscaled_message_p,
            ) = outputs
        else:
            model_out, self._value_out, h, c, *self._unscaled_message_p = outputs

        next_states = [h, c]
        if self.use_receiver_bias:
            next_states.extend(self.no_message_outputs[2:4])

        if self.use_inference_policy:
            if self.pm_type == "moving_avg":
                action_logits = model_out[..., : -self.message_size]
                unscaled_message_logits = model_out[..., -self.message_size :]
                avg_message_logits = tf.log(self._avg_message_p) - tf.log(
                    1 - self._avg_message_p
                )
                scaled_message_logits = unscaled_message_logits - avg_message_logits
                model_out = tf.keras.layers.Concatenate()(
                    [action_logits, scaled_message_logits]
                )
            elif self.pm_type == "hyper_nn":
                action_logits = model_out[..., : -self.message_size]
                unscaled_message_logits = model_out[..., -self.message_size :]
                scaled_message_logits = unscaled_message_logits - self._pm_logits
                model_out = tf.keras.layers.Concatenate()(
                    [action_logits, scaled_message_logits]
                )
            else:
                raise NotImplementedError("Wrong type for inference_policy")

        self._model_out = tf.reshape(model_out, [-1, self.num_outputs])
        return self._model_out, next_states

    def get_initial_state(self):
        """See parent class for docs"""
        init_state = [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

        if self.use_receiver_bias:
            # States will be different during forward pass with no message
            init_state += [
                np.zeros(self.cell_size, np.float32),
                np.zeros(self.cell_size, np.float32),
            ]

        return init_state

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

    def no_message_out(self):
        """Returns the model_out when message from other agent is ignored"""
        if self.use_receiver_bias:
            model_out = self.no_message_outputs[0]
            return tf.reshape(model_out, [-1, self.num_outputs])
        else:
            raise ValueError("Model doesn't use receiver bias")

    def unscaled_message_p(self):
        """Returns message probabilities before scaling by p(m)"""
        if self.use_inference_policy:
            return tf.reshape(self._unscaled_message_p[0], [-1, self.message_size])
        else:
            raise ValueError("Model doesn't use inference policy")

    def pm_logits(self):
        if self.use_inference_policy and self.pm_type == "hyper_nn":
            return self._pm_logits
        else:
            raise ValueError("Model doesn't use hyper_nn for p(m)")

    def set_avg_message_state(self, avg_message_state):
        """Returns ops that assign the new p(m) and optionally t"""
        ops = []
        if self.use_inference_policy and self.pm_type == "moving_avg":
            if self.ewma_momentum is None:
                prev_t = self._avg_message_t
                t = prev_t + avg_message_state[1]
                outputs = (
                    self._avg_message_p * prev_t
                    + avg_message_state[0] * avg_message_state[1]
                ) / t
                ops.append(tf.assign(self._avg_message_t, t))
            else:
                outputs = (
                    self.ewma_momentum * self._avg_message_p
                    + (1 - self.ewma_momentum) * avg_message_state[0]
                )

            ops.append(tf.assign(self._avg_message_p, outputs))
        return ops


class SenderModel(GridworldMaModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, "sender")


class ReceiverModel(GridworldMaModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, "receiver")
