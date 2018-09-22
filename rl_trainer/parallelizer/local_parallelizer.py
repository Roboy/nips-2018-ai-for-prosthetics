import random
from multiprocessing import Pool
from typing import Callable, NamedTuple

import gym
from typeguard import typechecked

from rl_trainer.agent import GymAgent
from rl_trainer.experiment import Experiment


class ExperimentConfig(NamedTuple):
    """
    The 'environment_constructor' is a function that takes an int seed
    and returns a gym.Env environment.
    """
    agent: GymAgent
    environment_constructor: Callable[[], gym.Env]
    episodes_per_experiment: int = 100


class LocalParallelizer:
    """
    This class is responsible for parallelizing multiple experiments
    with a single agent on a single computer.
    """

    @typechecked
    def __init__(self, num_processes: int):
        self._num_processes = num_processes
        self._config = None

    @typechecked
    def run_experiments_with_config(self, config: ExperimentConfig):
        self._config = config
        with Pool(processes=self._num_processes) as pool:
            seeds = [random.randint(0, 2e63) for _ in range(self._num_processes)]
            pool.map(func=self._run_experiment, iterable=seeds)
        print(f"All experiments of {self} completed.")

    def _run_experiment(self, seed: int):
        experiment = Experiment(
            agent=self._config.agent,
            env=self._config.environment_constructor(),
            num_episodes=self._config.episodes_per_experiment,
        )
        experiment.run(seed=seed)
