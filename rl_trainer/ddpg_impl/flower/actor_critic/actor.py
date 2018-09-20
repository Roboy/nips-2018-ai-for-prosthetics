import tflearn
import tensorflow as tf


class Actor(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self._action_bound = action_bound
        self._learning_rate = learning_rate
        self._tau = tau
        self._batch_size = batch_size

        self._state_ph, self._action_output = self.create_policy_network()

        self._network_params = tf.trainable_variables()

        self._target_net_state_ph, self._target_net_action_output = self.create_policy_network()

        self.target_network_params = tf.trainable_variables()[len(self._network_params):]

        # Op for periodically updating target network with online network
        # weights
        self._update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self._network_params[i], self._tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self._tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self._action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self._unnormalized_actor_gradients = tf.gradients(
            self._action_output, self._network_params, -self._action_gradient)
        self._actor_gradients = [tf.div(g, self._batch_size) for g in self._unnormalized_actor_gradients]

        self.optimize = tf.train.AdamOptimizer(self._learning_rate).apply_gradients(
            zip(self._actor_gradients, self._network_params))

        self.num_trainable_vars = len(self._network_params) + len(self.target_network_params)

    def create_policy_network(self):
        state_ph = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(state_ph, 64)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 64)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        action_output = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        action_output = tf.multiply(action_output, self._action_bound)
        return state_ph, action_output

    def train(self, states_batch, action_grads_batch):
        self.sess.run(self.optimize, feed_dict={
            self._state_ph: states_batch,
            self._action_gradient: action_grads_batch
        })

    def predict(self, states_batch):
        return self.sess.run(self._action_output, feed_dict={
            self._state_ph: states_batch
        })

    def predict_target(self, states_batch):
        return self.sess.run(self._target_net_action_output, feed_dict={
            self._target_net_state_ph: states_batch
        })

    def update_target_network(self):
        self.sess.run(self._update_target_network_params)
