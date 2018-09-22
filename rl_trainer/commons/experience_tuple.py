import numpy as np
from typing import NamedTuple, Collection

from typeguard import typechecked


class ExperienceTuple(NamedTuple):
    state_1: Collection[float]
    action: Collection[float]
    reward: float
    state_2: Collection[float]
    state_2_is_terminal: bool

    def __eq__(self, other) -> bool:
        return all([
            np.isclose(self.state_1, other.state_1).all(),
            np.isclose(self.action, other.action).all(),
            np.isclose(self.reward, other.reward).all(),
            np.isclose(self.state_2, other.state_2).all(),
            self.state_2_is_terminal == other.state_2_is_terminal,
        ])

    @staticmethod
    def mock(state_dim, action_dim):
        return ExperienceTuple(
            state_1=np.random.random(state_dim),
            action=np.random.random(action_dim),
            reward=np.random.random(),
            state_2=np.random.random(state_dim),
            state_2_is_terminal=False,
        )


class Episode(NamedTuple):
    experience_tuples: Collection[ExperienceTuple]


class ExperienceTupleBatch:
    """
    Allows to access the list of attributes without constructing
    them explicitly. For example: state_batch = batch.initial_states
    """

    @typechecked
    def __init__(self, experience_tuples: Collection[ExperienceTuple]):
        self.experience_tuples = experience_tuples
        self._init()

    def _init(self):
        gen = ((e.state_1, e.action, e.reward, e.state_2, e.state_2_is_terminal) for e in self.experience_tuples)
        (
            self.initial_states,
            self.actions,
            self.rewards,
            self.final_states,
            self.final_states_are_terminal,
        ) = zip(*gen)

    def __len__(self):
        return len(self.experience_tuples)
