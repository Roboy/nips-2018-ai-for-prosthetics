from typing import Sequence, List

from commons import ExperienceTuple, Episode


class ProstheticsEnvAgent:

    def act(self, state: Sequence[float]) -> List[float]:
        assert isinstance(state, list)
        assert len(state) == ExperienceTuple.DIM_STATE
        action = self._act(state)
        assert isinstance(action, list)
        assert len(action) == ExperienceTuple.DIM_ACTION
        return self._act(state)

    def _act(self, state: Sequence[float]) -> List[float]:
        raise NotImplementedError

    def train(self, episodes: Sequence[Episode]):
        for e in episodes:
            assert isinstance(e, Episode)
        self._train(episodes)

    def _train(self, episodes: Sequence[Episode]):
        raise NotImplementedError
