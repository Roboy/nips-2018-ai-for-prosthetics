import tflearn
import tensorflow as tf
from typeguard import typechecked
import numpy as np


class TFPolicyNetwork:

    @typechecked
    def __init__(self, sess: tf.Session, state_dim: int, action_dim: int,
                 action_bound: np.ndarray):
        assert len(action_bound) == action_dim
        self._sess = sess
        existing_vars = tf.trainable_variables()
        self._state_ph = tflearn.input_data(shape=[None, state_dim])
        self._action_output = self._construct_nn(action_bound, action_dim)
        self._variables = [var for var in tf.trainable_variables() if var not in existing_vars]

    def _construct_nn(self, action_bound, action_dim):
        net = tflearn.fully_connected(self._state_ph, 64)
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

    def predict(self, states_batch):
        return self._sess.run(self._action_output, feed_dict={
            self._state_ph: states_batch
        })


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

        self._online_nn = TFPolicyNetwork(sess, state_dim, action_dim, action_bound)
        self._critic_provided_action_grads = tf.placeholder(tf.float32, [None, action_dim])
        self._online_nn_train_op = self._setup_online_nn_train_op(
            learning_rate, batch_size, self._critic_provided_action_grads)

        self._target_nn = TFPolicyNetwork(sess, state_dim, action_dim, action_bound)
        self._target_nn_update_ops = self._setup_target_nn_update_ops(tau)

    @typechecked
    def _setup_online_nn_train_op(self, learning_rate: float, batch_size: int, critic_provided_action_grads):
        unnormalized_actor_grads = tf.gradients(
            ys=self._online_nn._action_output,
            xs=self._online_nn._variables,
            grad_ys=-critic_provided_action_grads
        )
        actor_gradients = [tf.div(g, batch_size) for g in unnormalized_actor_grads]
        adam = tf.train.AdamOptimizer(learning_rate)
        return adam.apply_gradients(
            grads_and_vars=zip(actor_gradients, self._online_nn._variables))

    @typechecked
    def _setup_target_nn_update_ops(self, tau: float):
        update_ops = []
        for running_var, target_var in zip(self._online_nn._variables, self._target_nn._variables):
            new_target_var = (1-tau)*target_var + tau*running_var
            update_ops.append(target_var.assign(new_target_var))
        return update_ops

    def online_nn_train(self, states_batch, action_grads_batch):
        self._sess.run(self._online_nn_train_op, feed_dict={
            self._online_nn._state_ph: states_batch,
            self._critic_provided_action_grads: action_grads_batch
        })

    def online_nn_predict(self, states_batch):
        return self._online_nn.predict(states_batch)

    def target_nn_predict(self, states_batch):
        return self._target_nn.predict(states_batch)

    def target_nn_update(self):
        self._sess.run(self._target_nn_update_ops)
