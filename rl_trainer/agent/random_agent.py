from typing import Collection

import gym

from rl_trainer.agent import GymAgent
from rl_trainer.commons import Episode


class RandomAgent(GymAgent):
    def __init__(self, action_space: gym.Space):
        self._action_space = action_space

    def act(self, state: Collection[float]) -> Collection[float]:
        return self._action_space.sample()

    def observe_episode(self, episode: Episode):
        pass
