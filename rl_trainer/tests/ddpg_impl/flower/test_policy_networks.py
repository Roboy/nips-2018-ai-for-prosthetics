import numpy as np
import pytest
import tensorflow as tf

from rl_trainer.commons import MockSpace
from rl_trainer.ddpg_impl.flower.actor_critic.policy_network import OnlinePolicyNetwork
from rl_trainer.ddpg_impl.flower.actor_critic.nn_baseclasses import OnlineNetwork, TargetNetwork

ACTION_DIM = 3
ACTION_SPACE = MockSpace(ACTION_DIM)
STATE_DIM = 2
STATE_SPACE = MockSpace(STATE_DIM)


@pytest.fixture(scope="module")
def tf_sess():
    with tf.Session(graph=tf.Graph()) as sess:
        return sess


@pytest.fixture(scope="module")
def online_policy_nn(tf_sess: tf.Session):
    nn = OnlinePolicyNetwork(action_bound=np.ones(ACTION_DIM), sess=tf_sess,
                             state_dim=STATE_DIM, action_dim=ACTION_DIM,
                             learning_rate=0.001, batch_size=64, action_space=ACTION_SPACE)
    assert isinstance(nn, OnlineNetwork)
    return nn


@pytest.fixture(scope="module")
def target_policy_nn(online_policy_nn: OnlinePolicyNetwork):
    target_nn = online_policy_nn.create_target_network(tau=0.5)
    assert isinstance(target_nn, TargetNetwork)
    return target_nn


def test_constructor_rejects_inconsistent_action_dim_input():
    with tf.Session(graph=tf.Graph()) as sess:
        with pytest.raises(AssertionError):
           OnlinePolicyNetwork(action_bound=np.ones(4), sess=sess,
                               state_dim=STATE_DIM, action_dim=ACTION_DIM,
                               learning_rate=0.001, batch_size=64, action_space=ACTION_SPACE)


def test_online_policy_nn_is_callable(online_policy_nn: OnlinePolicyNetwork,
                                      tf_sess: tf.Session):
    with tf_sess.graph.as_default():
        tf_sess.run(tf.global_variables_initializer())
    some_state = np.reshape(STATE_SPACE.sample(), (1, -1))
    action = online_policy_nn(s=some_state)[0]  # unpack tf batch shape
    for num in action:
        assert isinstance(num, np.float32), f"type of num: {type(num)}"


def test_online_policy_nn_train(online_policy_nn: OnlinePolicyNetwork, tf_sess: tf.Session):
    with tf_sess.graph.as_default():
        tf_sess.run(tf.global_variables_initializer())

    vars_before_train = [var.eval(tf_sess) for var in online_policy_nn._variables]

    batch_size = online_policy_nn.DEFAULT_BATCH_SIZE
    grads_a = np.array([ACTION_SPACE.sample() for _ in range(batch_size)])
    states = np.array([STATE_SPACE.sample() for _ in range(batch_size)])
    online_policy_nn.train(s=states, grads_a=grads_a)
    vars_after_train = [var.eval(tf_sess) for var in online_policy_nn._variables]

    for before, after, var in zip(vars_before_train, vars_after_train,
                                  online_policy_nn._variables):
        assert not np.allclose(before, after, 2e-8), f"var '{var.name}' is equal"
