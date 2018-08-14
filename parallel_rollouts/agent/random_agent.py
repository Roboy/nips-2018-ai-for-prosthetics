from typing import Sequence, List

import numpy as np

from agent import ProstheticsEnvAgent
from commons import ExperienceTuple


class RandomAgent(ProstheticsEnvAgent):
    def _act(self, state: Sequence[float]) -> List[float]:
        action_size = ExperienceTuple.DIM_ACTION
        return np.random.uniform(size=action_size).tolist()
