import numpy as np
import pytest
import tensorflow as tf

from rl_trainer.ddpg_impl.flower.actor_critic.actor import OnlinePolicyNetwork
from rl_trainer.ddpg_impl.flower.actor_critic.nn_templates import OnlineNetwork, TargetNetwork


def test_construction():
    with tf.Session() as sess:
        network = OnlinePolicyNetwork(action_bound=np.ones(3), sess=sess,
                                      state_dim=2, action_dim=3)
        assert isinstance(network, OnlineNetwork)


def test_constructor_rejects_inconsistent_input():
    with tf.Session() as sess:
        with pytest.raises(AssertionError):
           OnlinePolicyNetwork(action_bound=np.ones(4), sess=sess,
                               state_dim=2, action_dim=3)

def test_construction_of_target_nn():
    with tf.Session() as sess:
        online_nn = OnlinePolicyNetwork(action_bound=np.ones(3), sess=sess, state_dim=2, action_dim=3)
        target_nn = online_nn.create_target_network(tau=0.5)
        assert isinstance(target_nn, TargetNetwork)


def test_target_nn_update_op():
    with tf.Session() as sess:
        online_net = OnlinePolicyNetwork(action_bound=np.ones(3), sess=sess, state_dim=2, action_dim=3)
        target_net = online_net.create_target_network(tau=0.5)
        sess.run(tf.global_variables_initializer())

        vars_before_update = [var.eval(sess) for var in target_net._variables]
        target_net.update()
        vars_after_update = [var.eval(sess) for var in target_net._variables]

        for before, after in zip(vars_before_update, vars_after_update):
            assert not np.array_equal(before, after)