import tflearn
import tensorflow as tf
from typeguard import typechecked
import numpy as np


class TFPolicyNetwork:

    @typechecked
    def __init__(self, state_dim: int, action_dim: int, action_bound: np.ndarray):
        assert len(action_bound) == action_dim
        existing_vars = tf.trainable_variables()
        self.state_ph = tflearn.input_data(shape=[None, state_dim])
        self.action_output = self._construct_nn(action_bound, action_dim)
        self.variables = [var for var in tf.trainable_variables() if var not in existing_vars]

    def _construct_nn(self, action_bound, action_dim):
        net = tflearn.fully_connected(self.state_ph, 64)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 64)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        action_output = tflearn.fully_connected(
            net, action_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        return tf.multiply(action_output, action_bound)


class Actor:
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self._sess = sess

        self._online_nn = TFPolicyNetwork(state_dim, action_dim, action_bound)
        self._critic_provided_action_grads = tf.placeholder(tf.float32, [None, action_dim])
        self._online_nn_train_op = self._setup_online_nn_train_op(learning_rate, batch_size)

        self._target_nn = TFPolicyNetwork(state_dim, action_dim, action_bound)
        self._target_nn_update_ops = self._setup_target_nn_update_ops(tau)

        self.num_trainable_vars = len(self._target_nn.variables + self._online_nn.variables)

    @typechecked
    def _setup_online_nn_train_op(self, learning_rate: float, batch_size: int):
        unnormalized_actor_grads = tf.gradients(
            ys=self._online_nn.action_output,
            xs=self._online_nn.variables,
            grad_ys=-self._critic_provided_action_grads
        )
        actor_gradients = [tf.div(g, batch_size) for g in unnormalized_actor_grads]
        adam = tf.train.AdamOptimizer(learning_rate)
        return adam.apply_gradients(
            grads_and_vars=zip(actor_gradients, self._online_nn.variables))

    def train(self, states_batch, action_grads_batch):
        self._sess.run(self._online_nn_train_op, feed_dict={
            self._online_nn.state_ph: states_batch,
            self._critic_provided_action_grads: action_grads_batch
        })

    def predict(self, states_batch):
        return self._sess.run(self._online_nn.action_output, feed_dict={
            self._online_nn.state_ph: states_batch
        })

    def predict_target(self, states_batch):
        return self._sess.run(self._target_nn.action_output, feed_dict={
            self._target_nn.state_ph: states_batch
        })

    def update_target_network(self):
        self._sess.run(self._target_nn_update_ops)

    @typechecked
    def _setup_target_nn_update_ops(self, tau: float):
        update_ops = []
        for running_var, target_var in zip(self._online_nn.variables, self._target_nn.variables):
            new_target_var = (1-tau)*target_var + tau*running_var
            update_ops.append(target_var.assign(new_target_var))
        return update_ops
