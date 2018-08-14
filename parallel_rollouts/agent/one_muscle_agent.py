from typing import Sequence, List

import numpy as np

from agent import ProstheticsEnvAgent
from commons import Episode, ExperienceTuple


class OneMuscleAgent(ProstheticsEnvAgent):
    def __init__(self):
        self.muscle_active = np.random.randint(0, ExperienceTuple.DIM_ACTION)
        zeros = np.zeros(ExperienceTuple.DIM_ACTION)
        zeros[self.muscle_active] = 1
        self.activation = zeros.tolist()

    def _act(self, observation: Sequence[float]) -> List[float]:
        return list(self.activation)

    def _train(self, episodes: Sequence[Episode]):
        pass
