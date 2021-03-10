import logging

import numpy as np
from ray.rllib import SampleBatch, TFPolicy, Policy
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.dynamic_tf_policy import DynamicTFPolicy
from ray.rllib.utils.schedules import ConstantSchedule, PiecewiseSchedule
from ray.rllib.utils import override, try_import_tf
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.tracking_dict import UsageTrackingDict

try:
    from ray.util import log_once
except ImportError:
    from ray.rllib.utils.debug import log_once

tf = try_import_tf()
logger = logging.getLogger(__name__)


# Copied from DynamicTFPolicy since fake batch needs a proper time axis
class LossInitMixin:
    def __init__(self, sample_info):
        """
        Args:
            sample_info: Info dict which the env will return
        """
        self.sample_info = sample_info

    @override(DynamicTFPolicy)
    def _initialize_loss(self):
        def fake_array(tensor, none_shape):
            shape = tensor.shape.as_list()
            non_none_shape = [s for s in shape if s is not None]
            none_shape = none_shape if isinstance(none_shape, list) else [none_shape]
            shape = none_shape + non_none_shape
            return np.zeros(shape, dtype=tensor.dtype.as_numpy_dtype)

        T = self.config["model"]["max_seq_len"]
        B = self.config["train_batch_size"] // T
        dummy_batch = {
            SampleBatch.CUR_OBS: fake_array(self._obs_input, B * T),
            SampleBatch.NEXT_OBS: fake_array(self._obs_input, B * T),
            SampleBatch.DONES: np.array([False] * B * T, dtype=np.bool),
            SampleBatch.ACTIONS: fake_array(
                ModelCatalog.get_action_placeholder(self.action_space), B * T
            ),
            SampleBatch.REWARDS: np.array([0] * B * T, dtype=np.float32),
            SampleBatch.INFOS: np.array([self.sample_info] * B * T),
        }
        if self._obs_include_prev_action_reward:
            dummy_batch.update(
                {
                    SampleBatch.PREV_ACTIONS: fake_array(self._prev_action_input, B * T),
                    SampleBatch.PREV_REWARDS: fake_array(self._prev_reward_input, B * T),
                }
            )

        state_init = self.get_initial_state()
        state_batches = []
        for i, h in enumerate(state_init):
            dummy_batch["state_in_{}".format(i)] = np.repeat(
                np.expand_dims(h, 0), B * T, 0
            )
            dummy_batch["state_out_{}".format(i)] = np.repeat(
                np.expand_dims(h, 0), B * T, 0
            )
            state_batches.append(np.repeat(np.expand_dims(h, 0), B * T, 0))
        if state_init:
            dummy_batch["seq_lens"] = np.array([T] * B * T, dtype=np.int32)
        for k, v in self.extra_compute_action_fetches().items():
            dummy_batch[k] = fake_array(v, B * T)

        # postprocessing might depend on variable init, so run it first here
        self._sess.run(tf.global_variables_initializer())

        postprocessed_batch = self.postprocess_trajectory(SampleBatch(dummy_batch))

        # model forward pass for the loss (needed after postprocess to
        # overwrite any tensor state from that call)
        self.model(self._input_dict, self._state_in, self._seq_lens)

        if self._obs_include_prev_action_reward:
            train_batch = UsageTrackingDict(
                {
                    SampleBatch.PREV_ACTIONS: self._prev_action_input,
                    SampleBatch.PREV_REWARDS: self._prev_reward_input,
                    SampleBatch.CUR_OBS: self._obs_input,
                }
            )
            loss_inputs = [
                (SampleBatch.PREV_ACTIONS, self._prev_action_input),
                (SampleBatch.PREV_REWARDS, self._prev_reward_input),
                (SampleBatch.CUR_OBS, self._obs_input),
            ]
        else:
            train_batch = UsageTrackingDict({SampleBatch.CUR_OBS: self._obs_input})
            loss_inputs = [
                (SampleBatch.CUR_OBS, self._obs_input),
            ]

        for k, v in postprocessed_batch.items():
            if k in train_batch:
                continue
            elif v.dtype == np.object:
                continue  # can't handle arbitrary objects in TF
            elif k == "seq_lens" or k.startswith("state_in_"):
                continue
            shape = (None,) + v.shape[1:]
            dtype = np.float32 if v.dtype == np.float64 else v.dtype
            placeholder = tf.placeholder(dtype, shape=shape, name=k)
            train_batch[k] = placeholder

        for i, si in enumerate(self._state_in):
            train_batch["state_in_{}".format(i)] = si
        train_batch["seq_lens"] = self._seq_lens

        if log_once("loss_init"):
            logger.debug(
                "Initializing loss function with dummy input:\n\n{}\n".format(
                    summarize(train_batch)
                )
            )

        self._loss_input_dict = train_batch
        loss = self._do_loss_init(train_batch)
        for k in sorted(train_batch.accessed_keys):
            if k != "seq_lens" and not k.startswith("state_in_"):
                loss_inputs.append((k, train_batch[k]))

        TFPolicy._initialize_loss(self, loss, loss_inputs)
        if self._grad_stats_fn:
            self._stats_fetches.update(
                self._grad_stats_fn(self, train_batch, self._grads)
            )
        self._sess.run(tf.global_variables_initializer())


def _left_constant_interpolation(l, r, alpha):
    return l


# Modified version of Rllib's LearningRateSchedule
class LearningRateSchedule:
    """Mixin for TFPolicy that adds a learning rate schedule."""

    def __init__(self, lr, lr_schedule):
        self.cur_lr = tf.Variable(lr, name="lr", trainable=False)
        # self.cur_lr = tf.get_variable("lr", initializer=lr, trainable=False)
        if lr_schedule is None:
            self.lr_schedule = ConstantSchedule(lr, framework=None)
        else:
            self.lr_schedule = PiecewiseSchedule(
                lr_schedule,
                interpolation=_left_constant_interpolation,
                outside_value=lr_schedule[-1][-1],
                framework=None,
            )

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super(LearningRateSchedule, self).on_global_var_update(global_vars)
        self.cur_lr.load(
            self.lr_schedule.value(global_vars["timestep"]), session=self._sess
        )
