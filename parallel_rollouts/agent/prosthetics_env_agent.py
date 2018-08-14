from typing import Sequence, List

from commons import ExperienceTuple


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
