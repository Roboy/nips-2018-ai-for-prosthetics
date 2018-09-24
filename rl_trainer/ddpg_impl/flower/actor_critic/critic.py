import tensorflow as tf
import tflearn
from typeguard import typechecked


class TFQNetwork:

    @typechecked
    def __init__(self, sess: tf.Session, state_dim: int, action_dim: int):
        self._sess = sess
        existing_vars = tf.trainable_variables()
        self._state_ph = tflearn.input_data(shape=[None, state_dim])
        self._action_ph = tflearn.input_data(shape=[None, action_dim])
        self._q_value_pred = self._construct_nn()
        self._variables = [var for var in tf.trainable_variables() if var not in existing_vars]

    def _construct_nn(self) -> tf.Tensor:
        net = tflearn.fully_connected(self._state_ph, 64)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 64)
        t2 = tflearn.fully_connected(self._action_ph, 64)
        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(self._action_ph, t2.W) + t2.b, activation='relu')
        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        return tflearn.fully_connected(net, 1, weights_init=w_init)

    def predict(self, states_batch, actions_batch):
        return self._sess.run(self._q_value_pred, feed_dict={
            self._state_ph: states_batch,
            self._action_ph: actions_batch,
        })


class Critic:
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau):
        self.sess = sess

        self._online_nn = TFQNetwork(sess=sess, state_dim=state_dim, action_dim=action_dim)
        self._q_value_ph = tf.placeholder(tf.float32, [None, 1])
        self._online_nn_train_op = self._setup_online_nn_training_op(learning_rate, self._q_value_ph)

        self._target_nn = TFQNetwork(sess=sess, state_dim=state_dim, action_dim=action_dim)
        self._target_nn_update_ops = self._setup_target_nn_update_ops(tau=tau)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self._online_nn_action_grads = tf.gradients(ys=self._online_nn._q_value_pred,
                                                    xs=self._online_nn._action_ph)

    def _setup_online_nn_training_op(self, learning_rate, q_value_ph):
        loss = tflearn.mean_square(q_value_ph, self._online_nn._q_value_pred)
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)

    def _setup_target_nn_update_ops(self, tau):
        update_ops = []
        for running_var, target_var in zip(self._online_nn._variables, self._target_nn._variables):
            new_target_var = (1-tau)*target_var + tau*running_var
            update_ops.append(target_var.assign(new_target_var))
        return update_ops

    @staticmethod
    @typechecked
    def create_q_network(state_dim: int, action_dim: int):
        state_placeholder = tflearn.input_data(shape=[None, state_dim])
        action_placeholder = tflearn.input_data(shape=[None, action_dim])
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

    def online_nn_train(self, states_batch, actions_batch, q_values_batch):
        return self.sess.run([self._online_nn._q_value_pred, self._online_nn_train_op], feed_dict={
            self._online_nn._state_ph: states_batch,
            self._online_nn._action_ph: actions_batch,
            self._q_value_ph: q_values_batch
        })

    def online_nn_predict(self, states_batch, actions_batch):
        return self.sess.run(self._online_nn._q_value_pred, feed_dict={
            self._online_nn._state_ph: states_batch,
            self._online_nn._action_ph: actions_batch
        })

    def target_nn_predict(self, states_batch, actions_batch):
        return self._target_nn.predict(states_batch=states_batch,
                                       actions_batch=actions_batch)

    def online_nn_action_gradients(self, states_batch, actions_batch):
        return self.sess.run(self._online_nn_action_grads, feed_dict={
            self._online_nn._state_ph: states_batch,
            self._online_nn._action_ph: actions_batch
        })

    def target_nn_update(self):
        self.sess.run(self._target_nn_update_ops)
