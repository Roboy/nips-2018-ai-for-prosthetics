import tensorflow as tf
import tflearn
from overrides import overrides
from typeguard import typechecked
import numpy as np

from rl_trainer.ddpg_impl.flower.actor_critic.nn_templates import TensorFlowNetwork, TensorFlowTargetNetwork, \
    OnlineNetwork


class TensorFlowQNetwork(TensorFlowNetwork):

    @typechecked
    @overrides
    def _construct_nn(self, state_dim: int, action_dim: int) -> None:
        self._state_ph = tflearn.input_data(shape=[None, state_dim])
        self._action_ph = tflearn.input_data(shape=[None, action_dim])
        net = self._fc_layer_on_state_input()
        net = self._concat_action_input_to_net(net)
        self._q_value_pred = tflearn.fully_connected(
            net, 1, weights_init="truncated_normal", bias_init="truncated_normal")

    def _fc_layer_on_state_input(self):
        net = self._default_fc_layer(self._state_ph)
        net = tflearn.layers.normalization.batch_normalization(
            net, beta=np.random.uniform(0, 0.0001))
        net = tflearn.activations.relu(net)
        return net

    @typechecked
    def _default_fc_layer(self, input: tf.Tensor) -> tf.Tensor:
        return tflearn.fully_connected(incoming=input, n_units=64,
                                       bias_init="truncated_normal")

    def _concat_action_input_to_net(self, net):
        t1 = self._default_fc_layer(net)
        t2 = self._default_fc_layer(self._action_ph)
        concat = tf.matmul(net, t1.W) + tf.matmul(self._action_ph, t2.W) + t2.b
        net = tflearn.activation(concat, activation='relu')
        return net

    def predict_q(self, states_batch, actions_batch):
        return self._sess.run(self._q_value_pred, feed_dict={
            self._state_ph: states_batch,
            self._action_ph: actions_batch,
        })


class TensorFlowOnlineQNetwork(TensorFlowQNetwork, OnlineNetwork):

    @typechecked
    @overrides
    def create_target_network(self, tau: float) -> TensorFlowTargetNetwork:
        state_dim = self._state_ph.get_shape().as_list()[1]
        action_dim = self._action_ph.get_shape().as_list()[1]
        return TensorFlowTargetQNetwork(online_nn_vars=self._variables, tau=tau,
                                        sess=self._sess, state_dim=state_dim,
                                        action_dim=action_dim)


class TensorFlowTargetQNetwork(TensorFlowQNetwork, TensorFlowTargetNetwork):
    pass


class Critic:
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    @typechecked
    def __init__(self, sess: tf.Session, state_dim: int, action_dim: int,
                 learning_rate: float = 0.001, tau: float = 0.001):
        self._sess = sess

        self.online_nn = TensorFlowOnlineQNetwork(sess=sess, state_dim=state_dim, action_dim=action_dim)
        self._q_value_ph = tf.placeholder(tf.float32, [None, 1])
        self._online_nn_train_op = self._setup_online_nn_training_op(learning_rate, self._q_value_ph)

        self.target_nn = self.online_nn.create_target_network(tau=tau)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self._online_nn_action_grads = tf.gradients(ys=self.online_nn._q_value_pred,
                                                    xs=self.online_nn._action_ph)

    def _setup_online_nn_training_op(self, learning_rate, q_value_ph):
        loss = tflearn.mean_square(q_value_ph, self.online_nn._q_value_pred)
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)

    def online_nn_train(self, states_batch, actions_batch, q_values_batch) -> None:
        self._sess.run([self._online_nn_train_op], feed_dict={
            self.online_nn._state_ph: states_batch,
            self.online_nn._action_ph: actions_batch,
            self._q_value_ph: q_values_batch
        })

    def online_nn_action_gradients(self, states_batch, actions_batch):
        return self._sess.run(self._online_nn_action_grads, feed_dict={
            self.online_nn._state_ph: states_batch,
            self.online_nn._action_ph: actions_batch
        })
