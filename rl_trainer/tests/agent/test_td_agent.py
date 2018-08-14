import pytest

from rl_trainer.agent.value_estimator import MockEstimator
from rl_trainer.agent.prosthetics_env_agent import MockSpace
from rl_trainer.agent.td_agent import TDAgent

ACTION_SPACE = MockSpace(size=3)
STATE_SPACE = MockSpace(size=6)


@pytest.fixture
def td_agent():
    return TDAgent(
        action_space=ACTION_SPACE,
        state_space=STATE_SPACE,
        value_estimator=MockEstimator()
    )


def test_act(td_agent: TDAgent):
    state = STATE_SPACE.sample()
    action = td_agent.act(state)
    assert ACTION_SPACE.shape[0] == len(action)
