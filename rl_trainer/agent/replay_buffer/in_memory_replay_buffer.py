import random
from collections import deque
from typing import Collection

from typeguard import typechecked

from rl_trainer.commons import ExperienceTuple, ExperienceTupleBatch
from . import ReplayBuffer


class InMemoryReplayBuffer(ReplayBuffer):
    """
    The right side of the deque contains the most recent experiences
    """

    @typechecked
    def __init__(self, lower_size_limit: int = 64, seed: int = None,
                 batch_size: int = 64, buffer_size: int = 10000):
        if seed is not None:
            random.seed(seed)
        self._batch_size = batch_size
        self._lower_size_limit = lower_size_limit
        self._buffer_size = buffer_size
        self._buffer = deque()

    @typechecked
    def extend(self, exp_tuples: Collection[ExperienceTuple]) -> None:
        self._buffer.extend(exp_tuples)
        while len(self._buffer) > self._buffer_size:
            self._buffer.popleft()

    @typechecked
    def sample_batch(self, batch_size: int = None) -> ExperienceTupleBatch:
        batch_size = batch_size if batch_size is not None else self._batch_size
        assert self.has_sufficient_samples(), "The buffer can not yet provide samples."
        batch_size = min(batch_size, len(self._buffer))
        exp_tuples = random.sample(self._buffer, batch_size)
        return ExperienceTupleBatch(experience_tuples=exp_tuples)

    def clear(self):
        self._buffer.clear()

    def has_sufficient_samples(self) -> bool:
        return len(self._buffer) >= self._lower_size_limit
