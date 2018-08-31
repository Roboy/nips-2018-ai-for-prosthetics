from typing import Sequence, NamedTuple

import gym


class ExperienceTuple(NamedTuple):
    initial_state: Sequence[float]
    action: Sequence[float]
    reward: float
    final_state: Sequence[float]
    final_state_is_terminal: bool


class ExperienceTupleFactory:
    _DEFAULT_RANDOM_REWARD = 0.0

    def __init__(self, state_space: gym.Space, action_space: gym.Space):
        self._state_space = state_space
        self._action_space = action_space

    def new_tuple(
            self,
            initial_state: Sequence[float],
            action: Sequence[float],
            reward: float,
            final_state: Sequence[float],
            final_state_is_final: bool,
    ) -> ExperienceTuple:
        assert len(initial_state) == self._state_space.shape[0]
        assert len(action) == self._action_space.shape[0]
        assert len(final_state) == self._state_space.shape[0]
        assert isinstance(reward, float)
        assert isinstance(final_state_is_final, bool)

        return ExperienceTuple(
            initial_state=initial_state,
            action=action,
            reward=reward,
            final_state=final_state,
            final_state_is_terminal=final_state_is_final
        )

    def random_tuple(self) -> ExperienceTuple:
        return ExperienceTuple(
            initial_state=self._state_space.sample(),
            action=self._action_space.sample(),
            final_state=self._state_space.sample(),
            reward=self._DEFAULT_RANDOM_REWARD,
            final_state_is_terminal=False,
        )
