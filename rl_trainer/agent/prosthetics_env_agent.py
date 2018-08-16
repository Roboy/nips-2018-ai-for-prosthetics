from typing import Sequence, List

import gym

from rl_trainer.commons import Episode


class GymAgent:

    def __init__(self, action_space: gym.Space, state_space: gym.Space):
        self._state_space = state_space
        self._action_space = action_space

    def act(self, state: Sequence[float]) -> List[float]:
        assert isinstance(state, list)
        assert len(state) == self._state_space.shape[0]
        action = self._act(state)
        assert isinstance(action, list)
        assert len(action) == self._action_space.shape[0]
        return action

    def _act(self, state: Sequence[float]) -> List[float]:
        raise NotImplementedError

    def train(self, episodes: Sequence[Episode]):
        for e in episodes:
            assert isinstance(e, Episode)
        self._train(episodes)

    def _train(self, episodes: Sequence[Episode]):
        raise NotImplementedError


class MockSpace:
    def __init__(self, size: int):
        self.shape = (size,)

    def sample(self):
        return [0.5] * self.shape[0]
