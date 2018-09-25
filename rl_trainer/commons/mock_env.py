import numpy as np

import gym
from gym.spaces import Box
from typeguard import typechecked


class MockSpace(Box):

    def __init__(self, size: int):
        super(MockSpace, self).__init__(low=np.zeros(size), high=np.ones(size))
        self.high = np.ones(size)
        self.low = np.zeros(size)

    def sample(self):
        return np.random.uniform(self.low, self.high).tolist()

    def contains(self, x):
        return all(self.low <= x) and all(x <= self.high)


class MockEnvironment(gym.Env):
    action_space = MockSpace(2)
    observation_space = MockSpace(3)

    @typechecked
    def __init__(self, episode_length: int = 5):
        self._are_states_terminal = [False]*(episode_length-1) + [True]
        self._done = iter(self._are_states_terminal)

    def reset(self):
        self._done = iter(self._are_states_terminal)
        return self.observation_space.sample()

    def step(self, action):
        final_state = self.observation_space.sample()
        reward = 1
        done = next(self._done)
        info = None
        return final_state, reward, done, info

    def render(self, mode='human'):
        pass
