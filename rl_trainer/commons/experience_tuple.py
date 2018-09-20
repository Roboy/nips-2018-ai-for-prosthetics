import numpy as np
from typing import Sequence, NamedTuple


class ExperienceTuple(NamedTuple):
    initial_state: Sequence[float]
    action: Sequence[float]
    reward: float
    final_state: Sequence[float]
    final_state_is_terminal: bool

    def __eq__(self, other) -> bool:
        return all([
            np.isclose(self.initial_state, other.initial_state).all(),
            np.isclose(self.action, other.action).all(),
            np.isclose(self.reward, other.reward).all(),
            np.isclose(self.final_state, other.final_state).all(),
            self.final_state_is_terminal == other.final_state_is_terminal,
        ])


class ExperienceTupleBatch:
    """
    Allows to access the sequences of attributes withouth constructing
    them explicitly. For example: state_batch = batch.initial_states
    """

    def __init__(self, experience_tuples: Sequence[ExperienceTuple]):
        for tup in experience_tuples:
            assert isinstance(tup, ExperienceTuple)
        self.experience_tuples = experience_tuples
        self._init()

    def _init(self):
        gen = ((e.initial_state, e.action, e.reward, e.final_state, e.final_state_is_terminal) for e in self.experience_tuples)
        (
            self.initial_states,
            self.actions,
            self.rewards,
            self.final_states,
            self.final_states_are_terminal,
        ) = zip(*gen)

    def __len__(self):
        return len(self.experience_tuples)


def mock_experience_tuple(action_dim: int, state_dim: int) -> ExperienceTuple:
    return ExperienceTuple(
        initial_state=np.random.random(state_dim),
        action=np.random.random(action_dim),
        reward=np.random.random(),
        final_state=np.random.random(state_dim),
        final_state_is_terminal=False,
    )
