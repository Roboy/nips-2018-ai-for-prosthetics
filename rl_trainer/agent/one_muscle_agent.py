from typing import Sequence, List

import gym
import numpy as np

from rl_trainer.agent import GymAgent
from rl_trainer.commons import Episode


class OneMuscleAgent(GymAgent):

    def __init__(self, action_space: gym.Space, state_space: gym.Space):
        super().__init__(action_space, state_space)
        action_dim = action_space.shape[0]
        self.muscle_active = np.random.randint(0, action_dim)
        self._activation = np.zeros(action_dim)
        self._activation[self.muscle_active] = 1

    def _act(self, observation: Sequence[float]) -> List[float]:
        return np.multiply(self._activation, self._action_space.sample())

    def _train(self, episodes: Sequence[Episode]):
        pass
