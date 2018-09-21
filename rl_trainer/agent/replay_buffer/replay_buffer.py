from typing import Collection

from rl_trainer.commons import ExperienceTuple, ExperienceTupleBatch


class ReplayBuffer:
    def extend(self, exp_tuples: Collection[ExperienceTuple]) -> None:
        raise NotImplementedError

    def sample_batch(self, batch_size: int = None) -> ExperienceTupleBatch:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def can_provide_samples(self) -> bool:
        raise NotImplementedError
