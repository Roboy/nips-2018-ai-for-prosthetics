import pytest

from rl_trainer.ddpg_impl.flower.actor_critic.q_network import OnlineQNetwork, \
    TargetQNetwork
import tensorflow as tf
import numpy as np

from rl_trainer.ddpg_impl.flower.actor_critic.nn_baseclasses import TargetNetwork, \
    OnlineNetwork

TAU = 0.1


@pytest.fixture(scope="module")
def tf_session():
    with tf.Session(graph=tf.Graph()) as sess:
        yield sess


@pytest.fixture(scope="module")
def online_q_nn(tf_session: tf.Session):
    net = OnlineQNetwork(sess=tf_session, state_dim=2, action_dim=3)
    assert isinstance(net, OnlineNetwork)
    return net


@pytest.fixture(scope="module")
def target_q_nn(online_q_nn: OnlineQNetwork):
        target_net = online_q_nn.create_target_network(tau=TAU)
        assert isinstance(target_net, TargetNetwork)
        return target_net


def test_construction_of_target_nns():
    with tf.Session(graph=tf.Graph()) as sess:
        net1 = TargetQNetwork(sess=sess, state_dim=4, action_dim=4, online_nn_vars=[], tau=TAU)
        assert isinstance(net1, TargetNetwork)


def test_target_nn_update_op(target_q_nn: TargetQNetwork, tf_session: tf.Session):
    with tf_session.graph.as_default():
        tf_session.run(tf.global_variables_initializer())
    vars_before_update = [var.eval(tf_session) for var in target_q_nn._variables]
    target_q_nn.update()
    vars_after_update = [var.eval(tf_session) for var in target_q_nn._variables]

    for before, after in zip(vars_before_update, vars_after_update):
        if np.all(before == after) and (np.all(before == 0) or np.all(before == 1)):
            continue
        assert not np.array_equal(before, after)


def test_target_nn_update_op_doesnt_change_online_nn(tf_session: tf.Session,
                                                     online_q_nn: OnlineQNetwork,
                                                     target_q_nn: TargetQNetwork):
    with tf_session.graph.as_default():
        tf_session.run(tf.global_variables_initializer())
    online_vars_before_update = [var.eval(tf_session) for var in online_q_nn._variables]
    target_q_nn.update()
    online_vars_after_update = [var.eval(tf_session) for var in online_q_nn._variables]

    for before, after in zip(online_vars_before_update, online_vars_after_update):
        assert np.array_equal(before, after)


def test_q_network_has_prefixed_var_names(online_q_nn: OnlineQNetwork,
                                          target_q_nn: TargetQNetwork):
    expected_prefix = online_q_nn.__class__.__name__
    for var in online_q_nn._variables:
        assert var.name.startswith(expected_prefix), \
            f"'{var.name}' var name doesnt start with '{expected_prefix}'"

    expected_prefix = target_q_nn.__class__.__name__
    for var in target_q_nn._variables:
        assert var.name.startswith(expected_prefix), \
            f"'{var.name}' var name doesnt start with '{expected_prefix}'"
