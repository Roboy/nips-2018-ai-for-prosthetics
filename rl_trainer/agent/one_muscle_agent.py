from typing import Sequence, List

import gym
import numpy as np

from rl_trainer.agent import GymAgent
from rl_trainer.commons import Episode


class OneMuscleAgent(GymAgent):

    def __init__(self, action_space: gym.Space, state_space: gym.Space):
        super().__init__(action_space, state_space)
        dim_action = self._action_space.shape[0]
        self.muscle_active = np.random.randint(0, dim_action)
        zeros = np.zeros(dim_action)
        zeros[self.muscle_active] = 1
        self.activation = zeros.tolist()

    def _act(self, observation: Sequence[float]) -> List[float]:
        return list(self.activation)

    def _train(self, episodes: Sequence[Episode]):
        pass
