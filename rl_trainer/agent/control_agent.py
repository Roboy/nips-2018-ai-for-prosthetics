import numpy as np
from typing import Sequence

import gym

from agent import GymAgent
from agent.value_estimator import ValueEstimator
from commons import Episode

# TODO: Not finished
class ControlAgent(GymAgent):
    """Based on Sutton Books' 'Episodic Semi-gradient Sarsa' """
    _NUM_ACTIONS = 10

    def __init__(self, action_space: gym.Space, state_space: gym.Space,
                 value_estimator: ValueEstimator):
        super().__init__(action_space, state_space)
        self._value_estimator = value_estimator

    def _act(self, state):
        random_actions = []
        for _ in range(self._NUM_ACTIONS):
            state_action = np.hstack((self._action_space.sample(), state))
            random_actions.append(state_action)
        state_action_values = self._value_estimator.predict(random_actions)
        max_value_indices = np.where(state_action_values == np.max(state_action_values))[0]
        choice = np.random.choice(max_value_indices)
        return random_actions[choice]

    def _train(self, episodes: Sequence[Episode]):
        pass
