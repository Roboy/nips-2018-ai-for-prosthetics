import tensorflow as tf
from overrides import overrides
from typeguard import typechecked

from .nn_baseclasses import TensorFlowNetwork, TargetNetwork, OnlineNetwork
import numpy as np


class QNetwork(TensorFlowNetwork):
    """
    Input to the network is the state and action, output is Q(s,a).
    """

    @typechecked
    @overrides
    def _construct_nn(self, state_dim: int, action_dim: int) -> None:
        with self._sess.graph.as_default():
            self._state_ph = tf.placeholder(
                tf.float32, shape=[None, state_dim], name="state")
            self._action_ph = tf.placeholder(
                tf.float32, shape=[None, action_dim], name="action")
            net = self._dense_layer_on_state_input()
            net = self._concat_action_input_to_net(net)
            self._q_value_output = tf.layers.dense(
                inputs=net, units=1, bias_initializer=tf.truncated_normal_initializer)

    def _dense_layer_on_state_input(self):
        net = tf.layers.dense(inputs=self._state_ph, units=64,
                              bias_initializer=tf.truncated_normal_initializer)
        net = tf.layers.batch_normalization(inputs=net, training=True)
        net = tf.nn.relu(net)
        return net

    def _concat_action_input_to_net(self, net):
        t1 = tf.layers.dense(inputs=net, units=64,
                             bias_initializer=tf.truncated_normal_initializer)
        t2 = tf.layers.dense(inputs=self._action_ph, units=64,
                             bias_initializer=tf.truncated_normal_initializer)
        concat = tf.concat((t1, t2), axis=1)
        return tf.nn.relu(concat)

    def __call__(self, s, a):
        return self._sess.run(self._q_value_output, feed_dict={
            self._state_ph: s,
            self._action_ph: a,
        })


class TargetQNetwork(QNetwork, TargetNetwork):
    pass


class OnlineQNetwork(QNetwork, OnlineNetwork):

    @typechecked
    def __init__(self, sess: tf.Session, state_dim: int, action_dim: int,
                 learning_rate: float = 0.001):
        super(OnlineQNetwork, self).__init__(
            sess=sess, state_dim=state_dim, action_dim=action_dim)

        with self._sess.graph.as_default():
            # Get the gradient of the net w.r.t. the action.
            # For each action in the minibatch (i.e., for each x in xs),
            # this will sum up the gradients of each critic output in the minibatch
            # w.r.t. that action. Each output is independent of all
            # actions except for one.
            self._action_grads = tf.gradients(ys=self._q_value_output, xs=self._action_ph)
            self._q_value_ph = tf.placeholder(tf.float32, [None, 1], name="q_value")
            loss = tf.losses.mean_squared_error(self._q_value_ph, self._q_value_output)
            self._train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    @typechecked
    @overrides
    def create_target_network(self, tau: float) -> TargetQNetwork:
        state_dim = self._state_ph.get_shape().as_list()[1]
        action_dim = self._action_ph.get_shape().as_list()[1]
        return TargetQNetwork(online_nn_vars=self._variables, tau=tau,
                              sess=self._sess, state_dim=state_dim,
                              action_dim=action_dim)

    @typechecked
    def train(self, s: np.ndarray, a: np.ndarray, y_i: np.ndarray):
        self._sess.run(self._train_op, feed_dict={
            self._state_ph: s,
            self._action_ph: a,
            self._q_value_ph: y_i,
        })

    def grads_a(self, s, a):
        return self._sess.run(self._action_grads, feed_dict={
            self._state_ph: s,
            self._action_ph: a,
        })
