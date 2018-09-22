from rl_trainer.commons import MockSpace
from rl_trainer.agent import RandomAgent
from rl_trainer.agent import OneMuscleAgent

ACTION_SPACE = MockSpace(size=3)
STATE_SPACE = MockSpace(size=4)


def test_one_muscle_agent_constructor():
    OneMuscleAgent(action_space=ACTION_SPACE)


def test_random_agent_constructor():
    RandomAgent(action_space=ACTION_SPACE)


def test_agents_act():
    agents = [
        OneMuscleAgent(action_space=ACTION_SPACE),
        RandomAgent(action_space=ACTION_SPACE),
    ]
    state = STATE_SPACE.sample()
    for agent in agents:
        action = agent.act(state)
        assert len(action) == ACTION_SPACE.shape[0]
        assert ACTION_SPACE.contains(action)
