from typing import Sequence, NamedTuple

import gym


class ExperienceTuple(NamedTuple):
        initial_state: Sequence[float]
        action: Sequence[float]
        final_state: Sequence[float]
        reward: float


class ExperienceTupleFactory:
    _DEFAULT_RANDOM_REWARD = 0.0

    def __init__(self, state_space: gym.Space, action_space: gym.Space):
        self._state_space = state_space
        self._action_space = action_space

    def new_tuple(
            self,
            initial_state: Sequence[float],
            action: Sequence[float],
            final_state: Sequence[float],
            reward: float,
    ) -> ExperienceTuple:
        assert len(initial_state) == self._state_space.shape[0]
        assert len(action) == self._action_space.shape[0]
        assert len(final_state) == self._state_space.shape[0]
        assert isinstance(reward, float)
        return ExperienceTuple(initial_state, action, final_state, reward)

    def random_tuple(self) -> ExperienceTuple:
        return ExperienceTuple(
            initial_state=self._state_space.sample(),
            action=self._action_space.sample(),
            final_state=self._state_space.sample(),
            reward=self._DEFAULT_RANDOM_REWARD,
        )
