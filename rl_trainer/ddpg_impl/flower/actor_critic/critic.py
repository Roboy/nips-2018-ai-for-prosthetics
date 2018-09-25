import tensorflow as tf
import tflearn
from overrides import overrides
from typeguard import typechecked
import numpy as np

from .nn_baseclasses import TensorFlowNetwork, TargetNetwork, OnlineNetwork


class CriticNetwork(TensorFlowNetwork):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    @typechecked
    @overrides
    def _construct_nn(self, state_dim: int, action_dim: int) -> None:
        self._state_ph = tflearn.input_data(shape=[None, state_dim])
        self._action_ph = tflearn.input_data(shape=[None, action_dim])
        net = self._fc_layer_on_state_input()
        net = self._concat_action_input_to_net(net)
        self._q_value_output = tflearn.fully_connected(
            net, 1, bias_init="truncated_normal")

    def _fc_layer_on_state_input(self):
        net = self._default_fc_layer(self._state_ph)
        net = tflearn.layers.normalization.batch_normalization(
            net, beta=np.random.uniform(0, 0.0001))
        net = tflearn.activations.relu(net)
        return net

    @typechecked
    def _default_fc_layer(self, x: tf.Tensor) -> tf.Tensor:
        return tflearn.fully_connected(incoming=x, n_units=64,
                                       bias_init="truncated_normal")

    def _concat_action_input_to_net(self, net):
        t1 = self._default_fc_layer(net)
        t2 = self._default_fc_layer(self._action_ph)
        concat = tf.matmul(net, t1.W) + tf.matmul(self._action_ph, t2.W) + t2.b
        net = tflearn.activation(concat, activation='relu')
        return net

    def __call__(self, s, a):
        return self._sess.run(self._q_value_output, feed_dict={
            self._state_ph: s,
            self._action_ph: a,
        })


class TargetCriticNetwork(CriticNetwork, TargetNetwork):
    pass


class OnlineCriticNetwork(CriticNetwork, OnlineNetwork):

    @typechecked
    def __init__(self, sess: tf.Session, state_dim: int, action_dim: int,
                 learning_rate: float = 0.001):
        super(OnlineCriticNetwork, self).__init__(
            sess=sess, state_dim=state_dim, action_dim=action_dim)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self._action_grads = tf.gradients(ys=self._q_value_output, xs=self._action_ph)
        self._q_value_ph = tf.placeholder(tf.float32, [None, 1])
        loss = tflearn.mean_square(self._q_value_ph, self._q_value_output)
        self._train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    @typechecked
    @overrides
    def create_target_network(self, tau: float) -> TargetCriticNetwork:
        state_dim = self._state_ph.get_shape().as_list()[1]
        action_dim = self._action_ph.get_shape().as_list()[1]
        return TargetCriticNetwork(online_nn_vars=self._variables, tau=tau,
                                   sess=self._sess, state_dim=state_dim,
                                   action_dim=action_dim)

    def train(self, s, a, y_i):
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
