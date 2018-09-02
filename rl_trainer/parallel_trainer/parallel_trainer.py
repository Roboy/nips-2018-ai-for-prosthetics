from multiprocessing.pool import Pool, ThreadPool
from typing import List, Callable, NamedTuple

import gym

from rl_trainer.agent import GymAgent
from rl_trainer.commons import Episode
from rl_trainer.experiment import Experiment


class Input(NamedTuple):
    agent: GymAgent
    env: gym.Env
    episodes_per_experiment: int


def run_experiment(input: Input) -> List[Episode]:
    return Experiment(
        agent=input.agent,
        env=input.env,
        num_episodes=input.episodes_per_experiment,
    ).run()


class ParallelTrainer:
    """
    This class is responsible for evolving and training an agent. It
    orchestrates the parallelization of many individual experiments
    and collects the results.
    """

    def __init__(
            self,
            initial_agent: GymAgent,
            env_constructor: Callable,
            episodes_per_experiment: int,
            num_processes: int,
    ):
        self.current_agent = initial_agent
        self._env_constructor = env_constructor
        self._episodes_per_experiment = episodes_per_experiment
        self.episodes_history: List[Episode] = []
        self._num_processes = num_processes

    def training_step(self):
        episodes: List[Episode] = self._parallelize_experiments()
        print("Parallel experiment of {} complete.".format(self))
        self.episodes_history.extend(episodes)
        self.current_agent.train(episodes)
        print("Training of {} complete".format(self.current_agent))

    def _parallelize_experiments(self) -> List[Episode]:
        with ThreadPool(processes=self._num_processes) as pool:
            inputs = [self._experiment_input() for _ in range(self._num_processes)]
            workers_responses = pool.map(run_experiment, inputs)
            all_episodes = [episode for res in workers_responses for episode in res]
            return all_episodes

    def _experiment_input(self):
        return Input(
            agent=self.current_agent,
            env=self._env_constructor(),
            episodes_per_experiment=self._episodes_per_experiment,
        )
