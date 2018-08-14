from typing import List

from commons import Episode
from rollout import RollOutConfiguration, RollOut


class Parallelizer:
    def __init__(self, num_processes: int):
        self._num_processes = num_processes

    def launch_in_parallel(self, configuration: RollOutConfiguration) -> \
            List[Episode]:
        raise NotImplementedError


class MockParallelizer(Parallelizer):
    def launch_in_parallel(self, configuration: RollOutConfiguration) -> \
            List[Episode]:
        return RollOut(configuration).run()
