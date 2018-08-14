from typing import Sequence, List

from rl_trainer.agent import GymAgent
from rl_trainer.commons import Episode


class RandomAgent(GymAgent):
    def _act(self, state: Sequence[float]) -> List[float]:
        return list(self._action_space.sample())

    def _train(self, episodes: Sequence[Episode]):
        pass
