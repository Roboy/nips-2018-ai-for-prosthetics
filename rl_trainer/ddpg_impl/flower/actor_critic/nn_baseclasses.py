from typing import Collection

import tensorflow as tf
from typeguard import typechecked


class TensorFlowNetwork:
    @typechecked
    def __init__(self, sess: tf.Session, state_dim: int, action_dim: int):
        self._sess = sess
        existing_vars = tf.trainable_variables()
        self._construct_nn(state_dim, action_dim)
        self._variables: Collection[tf.Variable] = [
            var for var in tf.trainable_variables() if var not in existing_vars]

    @typechecked
    def _construct_nn(self, state_dim: int, action_dim: int) -> None:
        raise NotImplementedError


class TargetNetwork(TensorFlowNetwork):
    def __init__(self, online_nn_vars: Collection[tf.Variable], tau: float, **kwargs):
        super(TargetNetwork, self).__init__(**kwargs)
        self._target_nn_update_ops = self._setup_target_nn_update_ops(
            online_vars=online_nn_vars, tau=tau)

    def _setup_target_nn_update_ops(self, tau, online_vars: Collection[tf.Variable]):
        update_ops = []
        for online_var, target_var in zip(online_vars, self._variables):
            new_target_var = (1-tau)*target_var + tau*online_var
            update_ops.append(target_var.assign(new_target_var))
        return update_ops

    def update(self):
        self._sess.run(self._target_nn_update_ops)


class OnlineNetwork:
    def create_target_network(self, tau: float) -> TargetNetwork:
        raise NotImplementedError
