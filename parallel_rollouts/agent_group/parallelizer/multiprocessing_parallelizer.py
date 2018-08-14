from multiprocessing import Pool
from typing import List

from commons import Episode
from rollout import RollOutConfiguration, RollOut
from .parallelizer import Parallelizer


def do_rollout(configuration: RollOutConfiguration) -> List[Episode]:
    return RollOut(configuration).run()


class MultiProcessingParallelizer(Parallelizer):

    def _launch_in_parallel(self, configuration: RollOutConfiguration) -> \
            List[Episode]:
        with Pool(processes=self._num_processes) as pool:
            configurations = [configuration] * self._num_processes
            workers_responses = pool.map(do_rollout, configurations)
            flattened_episodes: List[Episode] = [episode for res in workers_responses for episode in res]
            return flattened_episodes
