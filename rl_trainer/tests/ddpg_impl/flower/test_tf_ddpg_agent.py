from unittest.mock import MagicMock

import pytest

from rl_trainer.commons import MockSpace
from rl_trainer.ddpg_impl.flower.actor_critic import TensorFlowDDPGAgent


@pytest.mark.integration
def test_tf_ddpg_agent():
    actor = MagicMock()
    critic = MagicMock()
    replay_buffer = MagicMock()
    TensorFlowDDPGAgent(state_dim=2, action_space=MockSpace(3), actor=actor,
                        critic=critic, replay_buffer=replay_buffer)
