from typing import Sequence, List


class ProstheticsEnvAgent:
    DIM_OBSERVATION = 158
    DIM_ACTION = 19

    def act(self, observation: Sequence[float]) -> List[float]:
        assert isinstance(observation, list)
        assert len(observation) == self.DIM_OBSERVATION
        action = self._act(observation)
        assert isinstance(action, list)
        assert len(action) == self.DIM_ACTION
        return self._act(observation)

    def _act(self, observation: Sequence[float]) -> List[float]:
        raise NotImplementedError
