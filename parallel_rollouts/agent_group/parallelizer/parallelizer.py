from typing import List

from commons import Episode
from rollout import RollOutConfiguration, RollOut


class Parallelizer:
    def __init__(self, num_processes: int):
        self._num_processes = num_processes

    def launch_in_parallel(self, configuration: RollOutConfiguration) -> \
            List[Episode]:
        assert isinstance(configuration, RollOutConfiguration)
        episodes = self._launch_in_parallel(configuration)
        for e in episodes:
            assert isinstance(e, Episode)
        return episodes

    def _launch_in_parallel(self, configuration: RollOutConfiguration):
        raise NotImplementedError


class MockParallelizer(Parallelizer):
    def _launch_in_parallel(self, configuration: RollOutConfiguration) -> \
            List[Episode]:
        return RollOut(configuration).run()
