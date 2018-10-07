import numpy as np
import pytest
import tensorflow as tf

from rl_trainer.commons import MockSpace
from rl_trainer.ddpg_impl.flower.actor_critic.actor import OnlineActorNetwork
from rl_trainer.ddpg_impl.flower.actor_critic.nn_baseclasses import OnlineNetwork, TargetNetwork

ACTION_DIM = 3
ACTION_SPACE = MockSpace(ACTION_DIM)
STATE_DIM = 2


@pytest.fixture(scope="module")
def online_policy_nn():
    with tf.Session(graph=tf.Graph()) as sess:
        nn = OnlineActorNetwork(action_bound=np.ones(ACTION_DIM), sess=sess,
                                state_dim=STATE_DIM, action_dim=ACTION_DIM,
                                learning_rate=0.001, batch_size=64, action_space=ACTION_SPACE)
        assert isinstance(nn, OnlineNetwork)
        return nn


def test_constructor_rejects_inconsistent_action_dim_input():
    with tf.Session(graph=tf.Graph()) as sess:
        with pytest.raises(AssertionError):
           OnlineActorNetwork(action_bound=np.ones(4), sess=sess,
                              state_dim=STATE_DIM, action_dim=ACTION_DIM,
                              learning_rate=0.001, batch_size=64, action_space=ACTION_SPACE)


def test_construction_of_target_nn(online_policy_nn: OnlineActorNetwork):
    target_nn = online_policy_nn.create_target_network(tau=0.5)
    assert isinstance(target_nn, TargetNetwork)
