import numpy as np
import pytest
import tensorflow as tf

from rl_trainer.ddpg_impl.flower.actor_critic.actor import OnlinePolicyNetwork
from rl_trainer.ddpg_impl.flower.actor_critic.nn_templates import OnlineNetwork, TargetNetwork


@pytest.fixture
def online_policy_nn():
    with tf.Session() as sess:
        nn = OnlinePolicyNetwork(action_bound=np.ones(3), sess=sess,
                                 state_dim=2, action_dim=3,
                                 learning_rate=0.001, batch_size=64)
        assert isinstance(nn, OnlineNetwork)
        return nn


def test_constructor_rejects_inconsistent_action_dim_input():
    with tf.Session() as sess:
        with pytest.raises(AssertionError):
           OnlinePolicyNetwork(action_bound=np.ones(4), sess=sess,
                               state_dim=2, action_dim=3,
                               learning_rate=0.001, batch_size=64)


def test_construction_of_target_nn(online_policy_nn: OnlinePolicyNetwork):
    target_nn = online_policy_nn.create_target_network(tau=0.5)
    assert isinstance(target_nn, TargetNetwork)
