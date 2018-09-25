from unittest.mock import MagicMock

import pytest

from rl_trainer.commons import MockSpace
from rl_trainer.ddpg_impl.flower.actor_critic import TensorFlowDDPGAgent


@pytest.mark.integration
def test_tf_ddpg_agent():
    actor_nn = MagicMock()
    critic_nn = MagicMock()
    replay_buffer = MagicMock()
    TensorFlowDDPGAgent(state_dim=2, action_space=MockSpace(3), critic_nn=critic_nn,
                        actor_nn=actor_nn, replay_buffer=replay_buffer, tf_model_saver=MagicMock())
