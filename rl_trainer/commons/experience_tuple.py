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


def mock_experience_tuple(action_dim: int, state_dim: int) -> ExperienceTuple:
    return ExperienceTuple(
        initial_state=np.random.random(state_dim),
        action=np.random.random(action_dim),
        reward=np.random.random(),
        final_state=np.random.random(state_dim),
        final_state_is_terminal=False,
    )
