from rl_trainer.ddpg_impl.flower.actor_critic.critic import OnlineQNetwork, \
    TargetQNetwork
import tensorflow as tf
import numpy as np

from rl_trainer.ddpg_impl.flower.actor_critic.nn_templates import TargetNetwork, \
    OnlineNetwork


def test_construction_of_online_nn():
    with tf.Session() as sess:
        net = OnlineQNetwork(sess=sess, state_dim=3, action_dim=4, learning_rate=0.001)
        assert isinstance(net, OnlineNetwork)


def test_create_target_nn():
    with tf.Session() as sess:
        online_net = OnlineQNetwork(sess=sess, state_dim=3,
                                    action_dim=4, learning_rate=0.001)
        target_net = online_net.create_target_network(tau=0.5)
        assert isinstance(target_net, TargetNetwork)


def test_construction_of_target_nns():
    with tf.Session() as sess:
        net1 = TargetQNetwork(sess=sess, state_dim=4, action_dim=4, online_nn_vars=[], tau=0.1)
        net2 = TargetQNetwork(sess=sess, state_dim=4, action_dim=4, online_nn_vars=[], tau=0.1)

        assert isinstance(net1, TargetNetwork)
        assert isinstance(net2, TargetNetwork)


def test_target_network_update_op():
    with tf.Session() as sess:
        _, target_nn = setup_nns(sess)

        vars_before_update = [var.eval(sess) for var in target_nn._variables]
        target_nn.update()
        vars_after_update = [var.eval(sess) for var in target_nn._variables]

        for before, after in zip(vars_before_update, vars_after_update):
            assert not np.array_equal(before, after)


def test_target_network_update_op_doesnt_change_online_net():
    with tf.Session() as sess:
        online_nn, target_nn = setup_nns(sess)

        vars_before_update = [var.eval(sess) for var in online_nn._variables]
        target_nn.update()
        vars_after_update = [var.eval(sess) for var in online_nn._variables]

        for before, after in zip(vars_before_update, vars_after_update):
            assert np.array_equal(before, after)


def setup_nns(sess):
    online_nn = OnlineQNetwork(sess=sess, state_dim=2, action_dim=2, learning_rate=0.001)
    target_nn = online_nn.create_target_network(tau=0.5)
    sess.run(tf.global_variables_initializer())
    return online_nn, target_nn
