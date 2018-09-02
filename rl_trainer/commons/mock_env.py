import random

import numpy as np

import gym


class MockSpace(gym.Space):

    def __init__(self, size: int):
        self.shape = (size,)
        self.high = np.ones(size)
        self.low = np.zeros(size)

    def sample(self):
        return np.random.uniform(self.low, self.high)

    def contains(self, x):
        return all(self.low <= x) and all(x <= self.high)


class MockEnvironment(gym.Env):
    action_space = MockSpace(2)
    state_space = MockSpace(3)
    _IS_STATE_TERMINAL = [False, False, True]

    def __init__(self):
        self._done = iter(self._IS_STATE_TERMINAL)

    def reset(self):
        self._done = iter(self._IS_STATE_TERMINAL)
        return self.state_space.sample()

    def step(self, action):
        final_state = self.state_space.sample()
        reward = 1
        done = next(self._done)
        info = None
        return final_state, reward, done, info

    def render(self, mode='human'):
        pass
