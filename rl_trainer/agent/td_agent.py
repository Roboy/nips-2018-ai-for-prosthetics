import gym
import numpy as np
from typing import Sequence, List

from rl_trainer.agent import GymAgent
from rl_trainer.agent.value_estimator import ValueEstimator
from rl_trainer.commons import Episode


class TDAgent(GymAgent):

    def __init__(self, action_space: gym.Space, state_space: gym.Space,
                 value_estimator: ValueEstimator):
        super().__init__(action_space, state_space)
        self._value_estimator = value_estimator

    def _act(self, state: Sequence[float]) -> List[float]:
        num_action_samples = 10
        many_actions = [self._action_space.sample() for _ in range(num_action_samples)]
        state_action_values = self._value_estimator.predict(many_actions)
        max_value_indices = np.where(state_action_values == np.max(state_action_values))[0]
        choice = np.random.choice(max_value_indices)
        return many_actions[choice]

    def _train(self, episodes: Sequence[Episode]):
        for episode in episodes:
            for exp_tuple in episode.experience_tuples:
                pass
