import tflearn
import tensorflow as tf
from overrides import overrides
from typeguard import typechecked
import numpy as np

from .nn_templates import TensorFlowNetwork, OnlineNetwork, TargetNetwork


class PolicyNetwork(TensorFlowNetwork):

    @typechecked
    def __init__(self, action_bound: np.ndarray, sess: tf.Session,
                 state_dim: int, action_dim: int, **kwargs):
        self._action_bound = action_bound
        super(PolicyNetwork, self).__init__(sess=sess, state_dim=state_dim,
                                            action_dim=action_dim, **kwargs)

    @typechecked
    @overrides
    def _construct_nn(self, state_dim: int, action_dim: int) -> None:
        assert len(self._action_bound) == action_dim
        self._state_ph = tflearn.input_data(shape=[None, state_dim])
        net = tflearn.fully_connected(self._state_ph, 64, bias_init="truncated_normal")
        net = tflearn.layers.normalization.batch_normalization(net, beta=np.random.uniform(0, 0.0001))
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 64, bias_init="truncated_normal")
        net = tflearn.layers.normalization.batch_normalization(net, beta=np.random.uniform(0, 0.0001))
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        action_output = tflearn.fully_connected(
            net, action_dim, activation='tanh', weights_init=w_init, bias_init="truncated_normal")
        # Scale output to -action_bound to action_bound
        self._action_output = tf.multiply(action_output, self._action_bound)

    def __call__(self, s):
        return self._sess.run(self._action_output, feed_dict={
            self._state_ph: s
        })


class TargetPolicyNetwork(PolicyNetwork, TargetNetwork):
    pass


class OnlinePolicyNetwork(PolicyNetwork, OnlineNetwork):

    def __init__(self, action_bound: np.ndarray, sess: tf.Session,
                 state_dim: int, action_dim: int, learning_rate: float, batch_size: int):
        super(OnlinePolicyNetwork, self).__init__(
            action_bound=action_bound, sess=sess,
            state_dim=state_dim, action_dim=action_dim)

        self._train_op = self._setup_train_op(
            action_dim=action_dim, batch_size=batch_size, learning_rate=learning_rate)

    @typechecked
    @overrides
    def create_target_network(self, tau: float) -> TargetPolicyNetwork:
        state_dim = self._state_ph.get_shape().as_list()[1]
        action_dim = self._action_output.get_shape().as_list()[1]
        return TargetPolicyNetwork(
            action_bound=self._action_bound, sess=self._sess, state_dim=state_dim,
            action_dim=action_dim, online_nn_vars=self._variables, tau=tau)

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


class Actor:
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    @typechecked
    def __init__(self, sess: tf.Session, state_dim: int, action_dim: int,
                 action_bound, batch_size: int = 64,
                 learning_rate: float = 1e-4, tau: float = 0.001):
        self._sess = sess

        self.μ = OnlinePolicyNetwork(sess=sess, state_dim=state_dim, action_dim=action_dim,
                                     action_bound=action_bound, learning_rate=learning_rate,
                                     batch_size=batch_size)
        self._critic_provided_action_grads = tf.placeholder(tf.float32, [None, action_dim])
        self._online_nn_train_op = self._setup_online_nn_train_op(
            learning_rate, batch_size, self._critic_provided_action_grads)

        self.μʹ = self.μ.create_target_network(tau=tau)

    @typechecked
    def _setup_online_nn_train_op(self, learning_rate: float, batch_size: int, critic_provided_action_grads):
        unnormalized_actor_grads = tf.gradients(
            ys=self.μ._action_output,
            xs=self.μ._variables,
            grad_ys=-critic_provided_action_grads
        )
        actor_gradients = [tf.div(g, batch_size) for g in unnormalized_actor_grads]
        adam = tf.train.AdamOptimizer(learning_rate)
        return adam.apply_gradients(
            grads_and_vars=zip(actor_gradients, self.μ._variables))

    def online_nn_train2(self, s, grads_a):
        self.μ.train(s=s, grads_a=grads_a)
