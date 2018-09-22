import pytest

from rl_trainer.agent.replay_buffer import InMemoryReplayBuffer
from rl_trainer.commons import ExperienceTupleBatch
from rl_trainer.commons.experience_tuple import mock_experience_tuple

BUFFER_SIZE = 10
LOWER_SIZE_LIMIT = 1
EXPERIENCE_TUPLE = mock_experience_tuple(action_dim=3, state_dim=4)


def test_construction():
    InMemoryReplayBuffer(buffer_size=BUFFER_SIZE,
                         lower_size_limit=LOWER_SIZE_LIMIT)


def test_lower_size_limit():
    buffer = InMemoryReplayBuffer(buffer_size=BUFFER_SIZE,
                                  lower_size_limit=LOWER_SIZE_LIMIT)
    assert not buffer.has_sufficient_samples()
    buffer.extend([EXPERIENCE_TUPLE])
    assert buffer.has_sufficient_samples()


def test_can_provide_samples():
    buffer = InMemoryReplayBuffer(buffer_size=BUFFER_SIZE,
                                  lower_size_limit=LOWER_SIZE_LIMIT)
    assert not buffer.has_sufficient_samples()
    with pytest.raises(AssertionError):
        buffer.sample_batch(3)


def test_can_sample_batch():
    buffer = InMemoryReplayBuffer(buffer_size=BUFFER_SIZE,
                                  lower_size_limit=LOWER_SIZE_LIMIT)
    buffer.extend([EXPERIENCE_TUPLE])
    buffer.extend([EXPERIENCE_TUPLE])

    batch = buffer.sample_batch(batch_size=2)
    assert isinstance(batch, ExperienceTupleBatch)
    assert len(batch) is 2
