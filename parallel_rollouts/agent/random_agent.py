from typing import Sequence, List

import numpy as np

from agent import ProstheticsEnvAgent


class RandomAgent(ProstheticsEnvAgent):
    def _act(self, observation: Sequence[float]) -> List[float]:
        return np.random.uniform(size=self.DIM_ACTION).tolist()
