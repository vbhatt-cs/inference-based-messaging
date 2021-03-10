import gym
import numpy as np
from ray.rllib import SampleBatch
from ray.rllib.agents.impala import ImpalaTrainer, vtrace
from ray.rllib.agents.impala.vtrace_policy import (
    VTraceTFPolicy,
    _make_time_major,
    BEHAVIOUR_LOGITS,
)
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.policy.policy import ACTION_LOGP
from ray.rllib.policy.tf_policy import EntropyCoeffSchedule
from ray.rllib.utils import try_import_tf

from ..utils.agent_utils import LossInitMixin, LearningRateSchedule
from ..utils.model_utils import CPCLayer

tf = try_import_tf()
CPC_INS = "cpc_ins"
CPC_PREDS = "cpc_preds"
WASTED_ACTS = "wasted_acts"


class CpcVTraceLoss:
    def __init__(
        self,
        actions,
        actions_logp,
        actions_entropy,
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
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0,
        use_cpc=True,
        cpc_ins=None,
        cpc_preds=None,
        cpc_coeff=10.0,
        **kwargs,
    ):
        """
        Adds CPC loss to VTraceLoss class
        Args:
            use_cpc: True if CPC loss should be added
            cpc_ins: Input encodings of CPC (Shape: [T, B, code_size]
            cpc_preds: Output encodings of CPC(Shape: [T, B, length, code_size]
            cpc_coeff: Coefficient for CPC loss
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

        # The summed weighted loss
        self.total_loss = (
            self.pi_loss + self.vf_loss * vf_loss_coeff - self.entropy * entropy_coeff
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


def build_cpc_vtrace_loss(policy, model, dist_class, train_batch):
    """Copied from build_vtrace_loss. Adds CPC loss"""

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
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(rewards)

    # Prepare actions for loss
    loss_actions = actions if is_multidiscrete else tf.expand_dims(actions, axis=1)

    use_cpc = policy.config["model"]["custom_options"]["use_cpc"]
    if use_cpc:
        cpc_ins = model.cpc_ins()
        cpc_preds = model.cpc_preds()
        cpc_config = policy.config["model"]["custom_options"]["cpc_opts"]
        cpc_config.update(
            dict(cpc_ins=make_time_major(cpc_ins), cpc_preds=make_time_major(cpc_preds))
        )
    else:
        cpc_config = {}

    # Inputs are reshaped from [B * T] => [T - 1, B] for V-trace calc.
    policy.loss = CpcVTraceLoss(
        actions=make_time_major(loss_actions, drop_last=False),
        actions_logp=make_time_major(action_dist.logp(actions), drop_last=False),
        actions_entropy=make_time_major(action_dist.multi_entropy(), drop_last=False),
        dones=dones,
        behaviour_action_logp=behaviour_action_logp,
        behaviour_logits=make_time_major(unpacked_behaviour_logits, drop_last=False),
        target_logits=make_time_major(unpacked_outputs, drop_last=False),
        discount=policy.config["gamma"],
        rewards=rewards,
        values=make_time_major(values, drop_last=False),
        bootstrap_value=bootstrap_value,
        dist_class=Categorical if is_multidiscrete else dist_class,
        model=model,
        valid_mask=make_time_major(mask, drop_last=False),
        config=policy.config,
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        entropy_coeff=policy.entropy_coeff,
        clip_rho_threshold=policy.config["vtrace_clip_rho_threshold"],
        clip_pg_rho_threshold=policy.config["vtrace_clip_pg_rho_threshold"],
        use_cpc=use_cpc,
        **cpc_config,
    )

    return policy.loss.total_loss


def stats(policy, train_batch):
    """Stats to save during training"""
    wasted_actions = tf.reduce_mean(train_batch[WASTED_ACTS])

    return {
        "cur_lr": tf.cast(policy.cur_lr, tf.float64),
        "policy_loss": policy.loss.pi_loss,
        "entropy": policy.loss.entropy,
        "vf_loss": policy.loss.vf_loss,
        "cpc_loss": policy.loss.cpc_loss,
        "wasted_actions": wasted_actions,
    }


def postprocess_trajectory(policy, sample_batch, other_agent_batches=None, episode=None):
    """Modified to store the percent of actions which didn't result in any movement"""
    infos = sample_batch.data[SampleBatch.INFOS]
    wasted_acts = np.array([e["waste"] for e in infos], dtype=np.float32)
    sample_batch.data.update({WASTED_ACTS: wasted_acts})
    return sample_batch


def setup_mixins(policy, obs_space, action_space, config):
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    EntropyCoeffSchedule.__init__(
        policy, config["entropy_coeff"], config["entropy_coeff_schedule"]
    )
    LossInitMixin.__init__(policy, {"waste": True})


CpcVTracePolicy = VTraceTFPolicy.with_updates(
    name="CpcVTracePolicy",
    loss_fn=build_cpc_vtrace_loss,
    stats_fn=stats,
    postprocess_fn=postprocess_trajectory,
    before_loss_init=setup_mixins,
    mixins=[LearningRateSchedule, EntropyCoeffSchedule, LossInitMixin],
)
ImpalaCPCSaTrainer = ImpalaTrainer.with_updates(
    name="impala_cpc_sa", default_policy=CpcVTracePolicy, get_policy_class=None
)
