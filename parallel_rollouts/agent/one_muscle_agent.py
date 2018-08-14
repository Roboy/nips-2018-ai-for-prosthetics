from typing import Sequence, List

import numpy as np

from agent import ProstheticsEnvAgent


class OneMuscleAgent(ProstheticsEnvAgent):
    def __init__(self):
        self.muscle_active = np.random.randint(0,self.DIM_ACTION)
        zeros = np.zeros(self.DIM_ACTION)
        zeros[self.muscle_active] = 1
        self.activation = zeros.tolist()
    def _act(self, observation: Sequence[float]) -> List[float]:
        return self.activation
