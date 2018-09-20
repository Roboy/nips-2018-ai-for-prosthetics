import tensorflow as tf
import tflearn


class Critic(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        self._state_ph, self._action_ph, self._q_value_pred = self.create_q_network()

        self._running_network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self._target_net_state_ph, self._target_net_action_ph, self._target_net_q_value_pred = self.create_q_network()

        self._target_network_params = tf.trainable_variables()[(len(self._running_network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self._update_target_network_params = \
            [self._target_network_params[i].assign(tf.multiply(self._running_network_params[i], self.tau) \
                                                   + tf.multiply(self._target_network_params[i], 1. - self.tau))
             for i in range(len(self._target_network_params))]

        # Network target (y_i)
        self._q_value_ph = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self._loss = tflearn.mean_square(self._q_value_ph, self._q_value_pred)
        self._optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self._loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self._action_grads = tf.gradients(self._q_value_pred, self._action_ph)

    def create_q_network(self):
        state_placeholder = tflearn.input_data(shape=[None, self.s_dim])
        action_placeholder = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(state_placeholder, 64)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 64)
        t2 = tflearn.fully_connected(action_placeholder, 64)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action_placeholder, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        q_value_pred = tflearn.fully_connected(net, 1, weights_init=w_init)
        return state_placeholder, action_placeholder, q_value_pred

    def train(self, states_batch, actions_batch, q_value_batch):
        return self.sess.run([self._q_value_pred, self._optimize], feed_dict={
            self._state_ph: states_batch,
            self._action_ph: actions_batch,
            self._q_value_ph: q_value_batch
        })

    def predict(self, states_batch, actions_batch):
        return self.sess.run(self._q_value_pred, feed_dict={
            self._state_ph: states_batch,
            self._action_ph: actions_batch
        })

    def predict_target(self, states_batch, actions_batch):
        return self.sess.run(self._target_net_q_value_pred, feed_dict={
            self._target_net_state_ph: states_batch,
            self._target_net_action_ph: actions_batch
        })

    def action_gradients(self, states_batch, actions_batch):
        return self.sess.run(self._action_grads, feed_dict={
            self._state_ph: states_batch,
            self._action_ph: actions_batch
        })

    def update_target_network(self):
        self.sess.run(self._update_target_network_params)
