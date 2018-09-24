from rl_trainer.ddpg_impl.flower.actor_critic.critic import TensorFlowOnlineQNetwork, \
    TensorFlowTargetQNetwork
import tensorflow as tf
import numpy as np

from rl_trainer.ddpg_impl.flower.actor_critic.nn_templates import TensorFlowTargetNetwork, \
    OnlineNetwork


def test_construction_of_online_nn():
    with tf.Session() as sess:
        net = TensorFlowOnlineQNetwork(sess=sess, state_dim=3, action_dim=4)
        assert isinstance(net, OnlineNetwork)


def test_create_target_nn():
    with tf.Session() as sess:
        online_net = TensorFlowOnlineQNetwork(sess=sess, state_dim=3, action_dim=4)
        target_net = online_net.create_target_network(tau=0.5)
        assert isinstance(target_net, TensorFlowTargetNetwork)


def test_construction_of_target_nns():
    with tf.Session() as sess:
        net1 = TensorFlowTargetQNetwork(sess=sess, state_dim=4, action_dim=4, online_nn_vars=[], tau=0.1)
        net2 = TensorFlowTargetQNetwork(sess=sess, state_dim=4, action_dim=4, online_nn_vars=[], tau=0.1)

        assert isinstance(net1, TensorFlowTargetNetwork)
        assert isinstance(net2, TensorFlowTargetNetwork)


def test_target_network_update_op():
    with tf.Session() as sess:

        target_net = setup_target_nn(sess)

        sess.run(tf.global_variables_initializer())

        vars_before_update = [var.eval(sess) for var in target_net._variables]
        target_net.update()
        vars_after_update = [var.eval(sess) for var in target_net._variables]

        for before, after in zip(vars_before_update, vars_after_update):
            assert not np.array_equal(before, after)


def setup_target_nn(sess):
    online_net = TensorFlowOnlineQNetwork(sess=sess, state_dim=2, action_dim=2)
    target_net = online_net.create_target_network(tau=0.5)
    return target_net
