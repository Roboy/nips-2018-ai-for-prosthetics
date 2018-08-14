from rl_trainer.agent.prosthetics_env_agent import MockSpace
from rl_trainer.agent import RandomAgent
from rl_trainer.agent import GymAgent
from rl_trainer.agent import OneMuscleAgent

ACTION_SPACE = MockSpace(size=3)
STATE_SPACE = MockSpace(size=4)

def test_agents_construction():
    constructors = [
        GymAgent,
        OneMuscleAgent,
        RandomAgent,
    ]
    for ctor in constructors:
        ctor(action_space=ACTION_SPACE, state_space=STATE_SPACE)


def test_agents_act():
    agents = [
        OneMuscleAgent(action_space=ACTION_SPACE, state_space=STATE_SPACE),
        RandomAgent(action_space=ACTION_SPACE, state_space=STATE_SPACE),
    ]
    state = STATE_SPACE.sample()
    for agent in agents:
        action = agent.act(state)
        assert len(action) == ACTION_SPACE.shape[0]
