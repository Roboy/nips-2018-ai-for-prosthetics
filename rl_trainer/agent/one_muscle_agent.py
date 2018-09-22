from typing import Collection

import gym
import numpy as np

from rl_trainer.agent import GymAgent
from rl_trainer.commons import Episode


class OneMuscleAgent(GymAgent):

    def __init__(self, action_space: gym.Space):
        self._action_space = action_space
        action_dim = action_space.shape[0]
        self.muscle_active = np.random.randint(0, action_dim)
        self._activation = np.zeros(action_dim)
        self._activation[self.muscle_active] = 1

    def act(self, state: Collection[float]) -> Collection[float]:
        """Use np.multiply in case the action sample is a list."""
        return np.multiply(self._activation, self._action_space.sample())

    def observe_episode(self, episode: Episode):
        pass
