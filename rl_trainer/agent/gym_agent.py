import numpy as np
from typing import Sequence

import gym

from rl_trainer.commons import Episode


class GymAgent:

    def __init__(self, action_space: gym.Space, state_space: gym.Space):
        self._state_space = state_space
        self._action_space = action_space

    def act(self, state):
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        action = self._act(state)
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        assert self._action_space.contains(action)
        return action

    def _act(self, state):
        raise NotImplementedError

    def train(self, episodes: Sequence[Episode]):
        for e in episodes:
            assert isinstance(e, Episode)
        self._train(episodes)

    def _train(self, episodes: Sequence[Episode]):
        raise NotImplementedError
