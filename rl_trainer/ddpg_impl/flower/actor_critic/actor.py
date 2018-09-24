import tflearn
import tensorflow as tf
from overrides import overrides
from typeguard import typechecked
import numpy as np

from rl_trainer.ddpg_impl.flower.actor_critic.nn_templates import OnlineNetwork, \
    TargetNetwork
from .nn_templates import TensorFlowNetwork


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

    def act(self, states_batch):
        return self._sess.run(self._action_output, feed_dict={
            self._state_ph: states_batch
        })


class TargetPolicyNetwork(PolicyNetwork, TargetNetwork):
    pass


class OnlinePolicyNetwork(PolicyNetwork, OnlineNetwork):

    @typechecked
    @overrides
    def create_target_network(self, tau: float) -> TargetNetwork:
        state_dim = self._state_ph.get_shape().as_list()[1]
        action_dim = self._action_output.get_shape().as_list()[1]
        return TargetPolicyNetwork(
            action_bound=self._action_bound, sess=self._sess, state_dim=state_dim,
            action_dim=action_dim, online_nn_vars=self._variables, tau=tau)


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

        self.online_nn = OnlinePolicyNetwork(sess=sess, state_dim=state_dim,
                action_dim=action_dim, action_bound=action_bound)
        self._critic_provided_action_grads = tf.placeholder(tf.float32, [None, action_dim])
        self._online_nn_train_op = self._setup_online_nn_train_op(
            learning_rate, batch_size, self._critic_provided_action_grads)

        self.target_nn = self.online_nn.create_target_network(tau=tau)

    @typechecked
    def _setup_online_nn_train_op(self, learning_rate: float, batch_size: int, critic_provided_action_grads):
        unnormalized_actor_grads = tf.gradients(
            ys=self.online_nn._action_output,
            xs=self.online_nn._variables,
            grad_ys=-critic_provided_action_grads
        )
        actor_gradients = [tf.div(g, batch_size) for g in unnormalized_actor_grads]
        adam = tf.train.AdamOptimizer(learning_rate)
        return adam.apply_gradients(
            grads_and_vars=zip(actor_gradients, self.online_nn._variables))

    def online_nn_train(self, states_batch, action_grads_batch):
        self._sess.run(self._online_nn_train_op, feed_dict={
            self.online_nn._state_ph: states_batch,
            self._critic_provided_action_grads: action_grads_batch
        })
