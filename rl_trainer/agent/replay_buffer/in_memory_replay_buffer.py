import random
from collections import deque

from rl_trainer.commons import ExperienceTuple, ExperienceTupleBatch
from . import ReplayBuffer


class InMemoryReplayBuffer(ReplayBuffer):
    """
    The right side of the deque contains the most recent experiences
    """

    def __init__(self, buffer_size: int, lower_size_limit: int, seed: int = None):
        assert isinstance(buffer_size, int) and isinstance(lower_size_limit, int)
        if seed is not None:
            random.seed(seed)
        self._lower_size_limit = lower_size_limit
        self._buffer_size = buffer_size
        self._buffer = deque()

    def add(self, exp_tuple: ExperienceTuple) -> None:
        assert isinstance(exp_tuple, ExperienceTuple)
        self._buffer.append(exp_tuple)
        if len(self._buffer) > self._buffer_size:
            self._buffer.popleft()

    def sample_batch(self, batch_size: int) -> ExperienceTupleBatch:
        assert isinstance(batch_size, int)
        assert self.can_provide_samples(), "The buffer can not yet provide samples."
        batch_size = min(batch_size, len(self._buffer))
        exp_tuples = random.sample(self._buffer, batch_size)
        return ExperienceTupleBatch(experience_tuples=exp_tuples)

    def clear(self):
        self._buffer.clear()

    def can_provide_samples(self) -> bool:
        return len(self._buffer) >= self._lower_size_limit