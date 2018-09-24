import numpy as np

from rl_trainer.commons import ExperienceTupleBatch, ExperienceTuple

EXPERIENCE_TUPLE = ExperienceTuple.mock(3, 4)
batch = ExperienceTupleBatch(experience_tuples=[EXPERIENCE_TUPLE, EXPERIENCE_TUPLE])


def test_experience_tuple_batch():
    assert len(batch) == 2


def test_states_1():
    assert len(batch.states_1) is 2
    for state in batch.states_1:
        assert np.array_equal(state, EXPERIENCE_TUPLE.state_1)


def test_actions():
    assert len(batch.actions) is 2
    for action in batch.actions:
        assert np.array_equal(action, EXPERIENCE_TUPLE.action)


def test_rewards():
    assert len(batch.rewards) is 2
    for reward in batch.rewards:
        assert reward == EXPERIENCE_TUPLE.reward


def test_states_2():
    assert len(batch.states_2) is 2
    for state in batch.states_2:
        assert np.array_equal(state, EXPERIENCE_TUPLE.state_2)


def test_states_2_are_terminal():
    assert len(batch.states_2_are_terminal) is 2
    for done in batch.states_2_are_terminal:
        assert done is EXPERIENCE_TUPLE.state_2_is_terminal
