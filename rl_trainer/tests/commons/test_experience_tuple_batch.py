import numpy as np

from rl_trainer.commons import ExperienceTupleBatch, ExperienceTuple

EXPERIENCE_TUPLE = ExperienceTuple.mock(3, 4)
batch = ExperienceTupleBatch(experience_tuples=[EXPERIENCE_TUPLE, EXPERIENCE_TUPLE])


def test_experience_tuple_batch():
    assert len(batch) == 2


def test_initial_states():
    assert len(batch.initial_states) is 2
    for state in batch.initial_states:
        assert np.array_equal(state, EXPERIENCE_TUPLE.state_1)


def test_actions():
    assert len(batch.actions) is 2
    for action in batch.actions:
        assert np.array_equal(action, EXPERIENCE_TUPLE.action)


def test_rewards():
    assert len(batch.rewards) is 2
    for reward in batch.rewards:
        assert reward == EXPERIENCE_TUPLE.reward


def test_final_states():
    assert len(batch.final_states) is 2
    for state in batch.final_states:
        assert np.array_equal(state, EXPERIENCE_TUPLE.state_2)


def test_final_states_are_terminal():
    assert len(batch.final_states_are_terminal) is 2
    for done in batch.final_states_are_terminal:
        assert done is EXPERIENCE_TUPLE.state_2_is_terminal
