import tflearn
import tensorflow as tf
from gym.spaces import Box
from overrides import overrides
from typeguard import typechecked
import numpy as np

from .nn_baseclasses import TensorFlowNetwork, OnlineNetwork, TargetNetwork


class ActorNetwork(TensorFlowNetwork):

    @typechecked
    def __init__(self, action_bound: np.ndarray, action_space: Box, sess: tf.Session,
                 state_dim: int, action_dim: int, **kwargs):
        self._action_space = action_space
        self._action_bound = action_bound
        super(ActorNetwork, self).__init__(sess=sess, state_dim=state_dim,
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
            self._state_ph = tflearn.input_data(shape=[None, state_dim])
            net = tflearn.fully_connected(self._state_ph, 64, bias_init="truncated_normal")
            net = tflearn.layers.normalization.batch_normalization(net, beta=np.random.uniform(0, 0.0001))
            net = tflearn.activations.relu(net)
            net = tflearn.fully_connected(net, 64, bias_init="truncated_normal")
            net = tflearn.layers.normalization.batch_normalization(net, beta=np.random.uniform(0, 0.0001))
            net = tflearn.activations.relu(net)
            # Final layer weights are init to Uniform[-3e-3, 3e-3]
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            normalized_output = tflearn.fully_connected(
                net, action_dim, activation='tanh', weights_init=w_init, bias_init="truncated_normal")
            self._action_output = self._rescale_output(normalized_output)

    def _rescale_output(self, normalized_output):
        slope = (self._action_space.high-self._action_space.low)/2
        bias = (self._action_space.high+self._action_space.low)/2
        return normalized_output*slope + bias

    def __call__(self, s):
        return self._sess.run(self._action_output, feed_dict={
            self._state_ph: s
        })


class TargetActorNetwork(ActorNetwork, TargetNetwork):
    pass


class OnlineActorNetwork(ActorNetwork, OnlineNetwork):

    def __init__(self, action_bound: np.ndarray, sess: tf.Session, action_space: Box,
                 state_dim: int, action_dim: int, learning_rate: float = 1e-4,
                 batch_size: int = 64):
        super(OnlineActorNetwork, self).__init__(
            action_bound=action_bound, sess=sess,
            state_dim=state_dim, action_dim=action_dim, action_space=action_space)
        with self._sess.graph.as_default():
            self._train_op = self._setup_train_op(
                action_dim=action_dim, batch_size=batch_size, learning_rate=learning_rate)

    @typechecked
    @overrides
    def create_target_network(self, tau: float) -> TargetActorNetwork:
        state_dim = self._state_ph.get_shape().as_list()[1]
        action_dim = self._action_output.get_shape().as_list()[1]
        return TargetActorNetwork(
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
