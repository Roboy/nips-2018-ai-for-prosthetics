import tensorflow as tf
from gym.spaces import Box
from overrides import overrides
from typeguard import typechecked
import numpy as np

from .nn_baseclasses import TensorFlowNetwork, OnlineNetwork, TargetNetwork


class PolicyNetwork(TensorFlowNetwork):

    @typechecked
    def __init__(self, action_bound: np.ndarray, action_space: Box, sess: tf.Session,
                 state_dim: int, action_dim: int, **kwargs):
        self._action_space = action_space
        self._action_bound = action_bound
        super(PolicyNetwork, self).__init__(sess=sess, state_dim=state_dim,
                                            action_dim=action_dim, **kwargs)

    @typechecked
    @overrides
    def _construct_nn(self, state_dim: int, action_dim: int) -> None:
        """
        The output layer activation is a tanh to keep the action
        between -action_bound and action_bound
        """
        assert len(self._action_bound) == action_dim
        with self._sess.graph.as_default():
            self._state_ph = tf.placeholder(tf.float32, shape=[None, state_dim], name="state")
            net = self._add_dense_batchnorm_relu(self._state_ph)
            net = self._add_dense_batchnorm_relu(net)
            final_layer_weights = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
            normalized_output = tf.layers.dense(
                inputs=net, units=action_dim, activation=tf.nn.tanh,
                kernel_initializer=final_layer_weights,
                bias_initializer=tf.truncated_normal_initializer)
            self._action_output = self._rescale_output(normalized_output)

    @staticmethod
    def _add_dense_batchnorm_relu(inputs):
        net = tf.layers.dense(inputs=inputs, units=64,
                              bias_initializer=tf.truncated_normal_initializer)
        net = tf.layers.batch_normalization(inputs=net, training=True)
        return tf.nn.relu(net)

    def _rescale_output(self, normalized_output):
        slope = (self._action_space.high-self._action_space.low)/2
        bias = (self._action_space.high+self._action_space.low)/2
        return normalized_output*slope + bias

    def __call__(self, s):
        return self._sess.run(self._action_output, feed_dict={
            self._state_ph: s
        })


class TargetPolicyNetwork(PolicyNetwork, TargetNetwork):
    pass


class OnlinePolicyNetwork(PolicyNetwork, OnlineNetwork):

    DEFAULT_BATCH_SIZE = 64

    def __init__(self, action_bound: np.ndarray, sess: tf.Session, action_space: Box,
                 state_dim: int, action_dim: int, learning_rate: float = 1e-4,
                 batch_size: int = None):
        super(OnlinePolicyNetwork, self).__init__(
            action_bound=action_bound, sess=sess,
            state_dim=state_dim, action_dim=action_dim, action_space=action_space)
        batch_size = batch_size if batch_size else self.DEFAULT_BATCH_SIZE
        with self._sess.graph.as_default():
            self._train_op = self._setup_train_op(
                action_dim=action_dim, batch_size=batch_size, learning_rate=learning_rate)

    @typechecked
    @overrides
    def create_target_network(self, tau: float) -> TargetPolicyNetwork:
        state_dim = self._state_ph.get_shape().as_list()[1]
        action_dim = self._action_output.get_shape().as_list()[1]
        return TargetPolicyNetwork(
            action_bound=self._action_bound, sess=self._sess, state_dim=state_dim,
            action_dim=action_dim, online_nn_vars=self._variables, tau=tau,
            action_space=self._action_space)

    def train(self, s, grads_a):
        self._sess.run(self._train_op, feed_dict={
            self._state_ph: s,
            self._critic_provided_action_grads: grads_a
        })

    @typechecked
    def _setup_train_op(self, action_dim: int, batch_size: int, learning_rate: float):
        self._critic_provided_action_grads = \
            tf.placeholder(dtype=tf.float32, shape=[None, action_dim])
        unnormalized_actor_grads = tf.gradients(
            ys=self._action_output,
            xs=self._variables,
            grad_ys=-self._critic_provided_action_grads,
        )
        actor_gradients = [tf.div(g, batch_size) for g in unnormalized_actor_grads]
        adam = tf.train.AdamOptimizer(learning_rate)
        return adam.apply_gradients(
            grads_and_vars=zip(actor_gradients, self._variables))
