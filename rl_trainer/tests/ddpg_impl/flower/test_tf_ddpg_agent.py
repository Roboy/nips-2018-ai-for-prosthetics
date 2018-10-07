import pytest

from rl_trainer.commons import MockSpace
from rl_trainer.ddpg_impl.flower.actor_critic import TensorFlowDDPGAgent

STATE_DIM = 2
STATE_SPACE = MockSpace(STATE_DIM)
ACTION_SPACE = MockSpace(3)


@pytest.fixture(scope="module")
def flower():
    return TensorFlowDDPGAgent(state_dim=STATE_DIM, action_space=ACTION_SPACE)


def test_tf_ddpg_agent_act(flower: TensorFlowDDPGAgent):
    action = flower.act(current_state=STATE_SPACE.sample())
    assert ACTION_SPACE.contains(action)
