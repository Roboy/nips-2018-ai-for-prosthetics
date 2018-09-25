from typing import Collection

from rl_trainer.commons import Episode


class GymAgent:
    def act(self, state: Collection[float]) -> Collection[float]:
        raise NotImplementedError

    def observe_episode(self, episode: Episode):
        raise NotImplementedError

    def set_seed(self, seed: int):
        raise NotImplementedError
