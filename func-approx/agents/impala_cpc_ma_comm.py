"""
Implementation of Eccles, Tom, et al. "Biases for Emergent Communication in Multi-agent
Reinforcement Learning." Advances in Neural Information Processing Systems. 2019.
"""


import gym
import numpy as np
from ray.rllib import SampleBatch
from ray.rllib.agents.impala import vtrace
from ray.rllib.agents.impala.vtrace_policy import _make_time_major, BEHAVIOUR_LOGITS
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import MultiActionDistribution, Categorical
from ray.rllib.policy.policy import ACTION_LOGP
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.tuple_actions import TupleActions

from .impala_cpc_sa import WASTED_ACTS, CpcVTracePolicy
from ..utils.model_utils import CPCLayer

tf = try_import_tf()
NO_MESSAGE_OBS = "no_message_obs"


class MessageActionDistribution(MultiActionDistribution):
    """Distribution for (action, message) tuple"""

    def __init__(self, inputs, model, action_space, name):
        child_dist = []
        input_lens = []
        for action in action_space.spaces:
            dist, action_size = ModelCatalog.get_action_dist(action, {})
            child_dist.append(dist)
            input_lens.append(action_size)
        super().__init__(inputs, model, action_space, child_dist, input_lens)
        with tf.variable_scope(name):
            self.entropy_list = [s.entropy() for s in self.child_distributions]

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        input_lens = []
        for action in action_space.spaces:
            dist, action_size = ModelCatalog.get_action_dist(action, {})
            input_lens.append(action_size)
        return sum(input_lens)

    def entropy(self):
        return self.entropy_list[0]

    def message_entropy(self):
        return self.entropy_list[-1]

    def mean_message_entropy(self):
        """Entropy of the mean message policy"""
        p_average = self.mean_message_p()
        logp_average = tf.log(p_average)
        return -tf.reduce_sum(p_average * logp_average)

    def mean_message_p(self):
        message_dist = self.child_distributions[-1]
        message_logits = message_dist.inputs
        p_bt = tf.nn.softmax(message_logits)
        p_average = tf.reduce_mean(p_bt, axis=0)
        return p_average

    def action_p(self):
        action_dist = self.child_distributions[0]
        action_logits = action_dist.inputs
        return tf.nn.softmax(action_logits)

    def action_logits(self):
        return self.child_distributions[0].inputs


class DeterministicMessageActionDistribution(MessageActionDistribution):
    """Distribution for (stochastic action, deterministic message) tuple"""

    def sample(self):
        return TupleActions(
            [
                self.child_distributions[0].sample(),
                self.child_distributions[1].deterministic_sample(),
            ]
        )

    def logp(self, x):
        split_indices = []
        for dist in self.child_distributions:
            if isinstance(dist, Categorical):
                split_indices.append(1)
            else:
                split_indices.append(tf.shape(dist.sample())[1])
        split_list = tf.split(x, split_indices, axis=1)
        for i, distribution in enumerate(self.child_distributions):
            # Remove extra categorical dimension
            if isinstance(distribution, Categorical):
                split_list[i] = tf.cast(tf.squeeze(split_list[i], axis=-1), tf.int32)

        log_action = self.child_distributions[0].logp(split_list[0])
        all_message_p = tf.nn.softmax(self.child_distributions[1].inputs)
        indices = tf.stack([tf.range(tf.shape(all_message_p)[0]), split_list[1]], axis=1)
        message_p = tf.gather_nd(all_message_p, indices)
        return log_action + message_p


class CommBiasLoss:
    def __init__(
        self,
        actions,
        actions_logp,
        actions_entropy,
        message_entropy,
        dones,
        behaviour_action_logp,
        behaviour_logits,
        target_logits,
        discount,
        rewards,
        values,
        bootstrap_value,
        dist_class,
        model,
        valid_mask,
        config,
        vf_loss_coeff=0.5,
        entropy_coeff=0.01,
        message_entropy_coeff=0.0,
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0,
        use_cpc=True,
        cpc_ins=None,
        cpc_preds=None,
        cpc_coeff=10.0,
        use_sender_bias=False,
        l_ps_lambda=3.0,
        entropy_target=1.0,
        average_message_entropy=None,
        sender_bias_coeff=0.1,
        use_receiver_bias=False,
        l_ce_coeff=0.001,
        l_pl_coeff=0.01,
        message_p=None,
        no_message_p=None,
        **kwargs,
    ):
        """
        See VTraceLoss class
        Args:
            use_cpc: True if CPC loss should be added
            cpc_ins: Input encodings of CPC (Shape: [T, B, code_size]
            cpc_preds: Output encodings of CPC(Shape: [T, B, length, code_size]
            cpc_coeff: Coefficient for CPC loss
            use_sender_bias: True if sender bias loss should be added
            l_ps_lambda:
        """
        # Compute vtrace on the CPU for better perf.
        with tf.device("/cpu:0"):
            self.vtrace_returns = vtrace.multi_from_logits(
                behaviour_action_log_probs=behaviour_action_logp,
                behaviour_policy_logits=behaviour_logits,
                target_policy_logits=target_logits,
                actions=tf.unstack(actions, axis=2),
                discounts=tf.to_float(~dones) * discount,
                rewards=rewards,
                values=values,
                bootstrap_value=bootstrap_value,
                dist_class=dist_class,
                model=model,
                clip_rho_threshold=tf.cast(clip_rho_threshold, tf.float32),
                clip_pg_rho_threshold=tf.cast(clip_pg_rho_threshold, tf.float32),
            )
            self.value_targets = self.vtrace_returns.vs

        # The policy gradients loss
        self.pi_loss = -tf.reduce_sum(
            tf.boolean_mask(actions_logp * self.vtrace_returns.pg_advantages, valid_mask)
        )

        # The baseline loss
        delta = tf.boolean_mask(values - self.vtrace_returns.vs, valid_mask)
        self.vf_loss = 0.5 * tf.reduce_sum(tf.square(delta))

        # The entropy loss
        self.entropy = tf.reduce_sum(tf.boolean_mask(actions_entropy, valid_mask))
        self.message_entropy = tf.reduce_sum(tf.boolean_mask(message_entropy, valid_mask))

        # The summed weighted loss
        self.total_loss = (
            self.pi_loss
            + self.vf_loss * vf_loss_coeff
            - self.entropy * entropy_coeff
            - self.message_entropy * message_entropy_coeff
        )

        if use_cpc:
            # CPC loss
            with tf.variable_scope("cpc_loss"):
                losses = []
                cpc_length = cpc_preds.shape.as_list()[2]
                T = tf.shape(cpc_preds)[0]
                # Scaling coeff to take mean over k
                scaling_coeff = tf.cast(
                    tf.reverse(tf.minimum(tf.range(1, T - 1 + 1), cpc_length), axis=[0]),
                    dtype=tf.float32,
                )
                for k in range(1, cpc_length + 1):
                    loss = CPCLayer(k, name=f"cpc_{k}")([cpc_ins, cpc_preds[:, :, k - 1]])
                    losses.append(tf.reduce_sum(loss / scaling_coeff[: T - k]))
                self.cpc_loss = tf.reduce_sum(tf.stack(losses), name=f"cpc_loss")
                self.total_loss += self.cpc_loss * cpc_coeff
        else:
            self.cpc_loss = tf.constant(np.nan)

        if use_sender_bias:
            # Sender bias loss
            with tf.variable_scope("sender_bias"):
                self.average_message_entropy = average_message_entropy
                self.sender_bias_loss = (
                    tf.reduce_sum(l_ps_lambda * (message_entropy - entropy_target) ** 2)
                    - average_message_entropy
                )
                self.total_loss += self.sender_bias_loss * sender_bias_coeff
        else:
            self.average_message_entropy = tf.constant(np.nan)
            self.sender_bias_loss = tf.constant(np.nan)

        if use_receiver_bias:
            # Receiver bias loss
            with tf.variable_scope("receiver_bias"):
                self.l_ce = -tf.reduce_sum(
                    tf.stop_gradient(message_p) * tf.log(no_message_p)
                )
                self.l_pl = tf.reduce_sum(
                    tf.abs(message_p - tf.stop_gradient(no_message_p))
                )
                self.total_loss += self.l_ce * l_ce_coeff - self.l_pl * l_pl_coeff
        else:
            self.l_ce = tf.constant(np.nan)
            self.l_pl = tf.constant(np.nan)


def build_ma_comm_loss(policy, model, dist_class, train_batch):
    """
    Copied from build_vtrace_loss. Adds CPC loss, comm biases and/or modifications for
    inference based messaging
    """

    def make_time_major(*args, **kw):
        return _make_time_major(policy, train_batch.get("seq_lens"), *args, **kw)

    actions = train_batch[SampleBatch.ACTIONS]
    dones = make_time_major(train_batch[SampleBatch.DONES], drop_last=False)
    rewards = make_time_major(train_batch[SampleBatch.REWARDS], drop_last=False)

    completed = tf.to_float(~dones)[-1]
    next_train_batch = {
        SampleBatch.CUR_OBS: make_time_major(train_batch[SampleBatch.NEXT_OBS])[-1],
        SampleBatch.PREV_ACTIONS: make_time_major(train_batch[SampleBatch.ACTIONS])[-1],
        SampleBatch.PREV_REWARDS: make_time_major(train_batch[SampleBatch.REWARDS])[-1],
        "seq_lens": tf.ones_like(train_batch["seq_lens"]),
    }
    i = 0
    while f"state_out_{i}" in train_batch:
        next_train_batch[f"state_in_{i}"] = make_time_major(
            train_batch[f"state_out_{i}"]
        )[-1]
        i += 1

    next_model_out, _ = model.from_batch(next_train_batch)
    next_values = model.value_function()
    bootstrap_value = tf.multiply(next_values, completed, name="bootstrap_value")

    model_out, _ = model.from_batch(train_batch)
    action_dist = dist_class(model_out, model)

    if isinstance(policy.action_space, gym.spaces.Discrete):
        is_multidiscrete = False
        output_hidden_shape = [policy.action_space.n]
    elif isinstance(policy.action_space, gym.spaces.multi_discrete.MultiDiscrete):
        is_multidiscrete = True
        output_hidden_shape = policy.action_space.nvec.astype(np.int32)
    else:
        is_multidiscrete = False
        output_hidden_shape = 1

    behaviour_action_logp = make_time_major(train_batch[ACTION_LOGP], drop_last=False)
    behaviour_logits = train_batch[BEHAVIOUR_LOGITS]
    unpacked_behaviour_logits = tf.split(behaviour_logits, output_hidden_shape, axis=1)
    unpacked_outputs = tf.split(model_out, output_hidden_shape, axis=1)
    values = model.value_function()

    if policy.is_recurrent():
        max_seq_len = tf.reduce_max(train_batch["seq_lens"]) - 1
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(rewards)

    # Prepare actions for loss
    loss_actions = actions if is_multidiscrete else tf.expand_dims(actions, axis=1)

    custom_opts = policy.config["model"]["custom_options"]
    use_cpc = custom_opts["use_cpc"]
    if use_cpc:
        cpc_ins = model.cpc_ins()
        cpc_preds = model.cpc_preds()
        cpc_config = custom_opts["cpc_opts"]
        cpc_config.update(
            dict(cpc_ins=make_time_major(cpc_ins), cpc_preds=make_time_major(cpc_preds))
        )
    else:
        cpc_config = {}

    use_sender_bias = custom_opts["use_sender_bias"]
    if use_sender_bias:
        size = tf.cast(tf.shape(actions)[0], tf.float32)
        sender_bias_config = custom_opts["sender_bias_opts"]
        sender_bias_config.update(
            {"average_message_entropy": action_dist.mean_message_entropy() * size}
        )
    else:
        sender_bias_config = {}

    use_receiver_bias = custom_opts["use_receiver_bias"]
    if use_receiver_bias:
        no_message_model_out = model.no_message_out()
        no_message_action_dist = dist_class(no_message_model_out, model)
        receiver_bias_config = dict(
            message_p=action_dist.action_p(),
            no_message_p=no_message_action_dist.action_p(),
            **custom_opts["receiver_bias_opts"],
        )
    else:
        receiver_bias_config = {}

    use_inference_policy = custom_opts["use_inference_policy"]
    if use_inference_policy:
        if custom_opts["inference_policy_opts"]["type"] == "moving_avg":
            # Update the moving average based on the rollout data
            # policy._avg_message_state used in build_apply_op() to store the curent
            # average for the next rollout
            ewma_momentum = custom_opts["inference_policy_opts"]["ewma_momentum"]
            unscaled_message_p = model.unscaled_message_p()
            policy._avg_message_state = [tf.reduce_mean(unscaled_message_p, axis=0)]
            if ewma_momentum is None:
                policy._avg_message_state += [
                    tf.cast(tf.shape(unscaled_message_p)[0], dtype=tf.float32)
                ]
        elif custom_opts["inference_policy_opts"]["type"] == "hyper_nn":
            # Find true p(m) for training the p(m) estimator (not used in the paper)
            # policy._pm_loss added to other losses (no scaling required since the p(m)
            # estimator and policy/value networks are independent
            pm_logits = model.pm_logits()
            unscaled_message_p = model.unscaled_message_p()
            pm_true = tf.reduce_mean(unscaled_message_p, axis=0)
            policy._pm_loss = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(labels=pm_true, logits=pm_logits)
            )
        else:
            raise NotImplementedError("Wrong type for inference_policy")

    # Inputs are reshaped from [B * T] => [T - 1, B] for V-trace calc.
    policy.loss = CommBiasLoss(
        actions=make_time_major(loss_actions, drop_last=False),
        actions_logp=make_time_major(action_dist.logp(actions), drop_last=False),
        actions_entropy=make_time_major(action_dist.multi_entropy(), drop_last=False),
        message_entropy=make_time_major(action_dist.message_entropy(), drop_last=False),
        dones=dones,
        behaviour_action_logp=behaviour_action_logp,
        behaviour_logits=make_time_major(unpacked_behaviour_logits, drop_last=False),
        target_logits=make_time_major(unpacked_outputs, drop_last=False),
        discount=policy.config["gamma"],
        rewards=rewards,
        values=make_time_major(values, drop_last=False),
        bootstrap_value=bootstrap_value,
        dist_class=dist_class,
        model=model,
        valid_mask=make_time_major(mask, drop_last=False),
        config=policy.config,
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        entropy_coeff=policy.entropy_coeff,
        message_entropy_coeff=policy.config["model"]["custom_options"][
            "message_entropy_coeff"
        ],
        clip_rho_threshold=policy.config["vtrace_clip_rho_threshold"],
        clip_pg_rho_threshold=policy.config["vtrace_clip_pg_rho_threshold"],
        use_cpc=use_cpc,
        **cpc_config,
        use_sender_bias=use_sender_bias,
        **sender_bias_config,
        use_receiver_bias=use_receiver_bias,
        **receiver_bias_config,
    )

    if hasattr(policy, "_pm_loss"):
        return policy.loss.total_loss + policy._pm_loss
    else:
        return policy.loss.total_loss


def stats(policy, train_batch):
    """Stats to save during training"""
    wasted_actions = tf.reduce_mean(train_batch[WASTED_ACTS])

    core_stats = {
        "cur_lr": tf.cast(policy.cur_lr, tf.float64),
        "policy_loss": policy.loss.pi_loss,
        "entropy": policy.loss.entropy,
        "message_entropy": policy.loss.message_entropy,
        "vf_loss": policy.loss.vf_loss,
        "cpc_loss": policy.loss.cpc_loss,
        "average_message_entropy": policy.loss.average_message_entropy,
        "sender_bias_loss": policy.loss.sender_bias_loss,
        "l_ce": policy.loss.l_ce,
        "l_pl": policy.loss.l_pl,
        "wasted_actions": wasted_actions,
    }

    if hasattr(policy.model, "_avg_message_p"):
        core_stats["avg_message_p"] = policy.model._avg_message_p
    if hasattr(policy, "_pm_loss"):
        core_stats["pm_loss"] = policy._pm_loss

    return core_stats


def build_apply_op(policy, optimizer, grads_and_vars):
    """
    Override for custom gradient apply computation. Only change from the original policy
    is setting the average message state in the case of inference based messaging.
    """
    ops = [
        optimizer.apply_gradients(
            policy._grads_and_vars, global_step=tf.train.get_or_create_global_step()
        )
    ]
    if hasattr(policy, "_avg_message_state") and policy._avg_message_state is not None:
        # Ops to update p(m) and optionally t for message probability scaling
        ops.extend(policy.model.set_avg_message_state(policy._avg_message_state))
    return ops


CommPolicy = CpcVTracePolicy.with_updates(
    loss_fn=build_ma_comm_loss, stats_fn=stats, apply_gradients_fn=build_apply_op
)

SenderPolicy = CommPolicy.with_updates(name="Sender")
ReceiverPolicy = CommPolicy.with_updates(name="Receiver")
