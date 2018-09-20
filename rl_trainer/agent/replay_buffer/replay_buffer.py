from rl_trainer.commons import ExperienceTuple, ExperienceTupleBatch


class ReplayBuffer:
    def add(self, exp_tuple: ExperienceTuple) -> None:
        raise NotImplementedError

    def sample_batch(self, batch_size: int) -> ExperienceTupleBatch:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def can_provide_samples(self) -> bool:
        raise NotImplementedError
