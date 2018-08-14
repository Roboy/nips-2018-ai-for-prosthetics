from multiprocessing.pool import Pool, ThreadPool
from typing import List, Callable, NamedTuple

import gym

from rl_trainer.agent import GymAgent
from rl_trainer.commons import Episode
from rl_trainer.interaction import IndependentInteraction


class Input(NamedTuple):
    agent: GymAgent
    env: gym.Env
    episodes_per_interaction: int


def run_interaction(input: Input) -> List[Episode]:
    return IndependentInteraction(
        agent=input.agent,
        env=input.env,
        num_episodes=input.episodes_per_interaction,
    ).run()


class ParallelTrainer:
    """
    This class is responsible for evolving and training an agent. It
    orchestrates the parallelization of many individual independent
    interactions and collects the results.
    """

    def __init__(
            self,
            initial_agent: GymAgent,
            env_constructor: Callable,
            episodes_per_interaction: int,
            num_processes: int,
    ):
        self.current_agent = initial_agent
        self._env_constructor = env_constructor
        self._episodes_per_interaction = episodes_per_interaction
        self.episodes_history: List[Episode] = []
        self._num_processes = num_processes

    def training_step(self):
        episodes: List[Episode] = self._parallelize_interactions()
        print("Parallel interaction of {} complete.".format(self))
        self.episodes_history.extend(episodes)
        self.current_agent.train(episodes)
        print("Training of {} complete".format(self.current_agent))

    def _parallelize_interactions(self) -> List[Episode]:
        with ThreadPool(processes=self._num_processes) as pool:
            interactions = [self._interaction_input() for _ in range(self._num_processes)]
            workers_responses = pool.map(run_interaction, interactions)
            all_episodes = [episode for res in workers_responses for episode in res]
            return all_episodes

    def _interaction_input(self):
        return Input(
            agent=self.current_agent,
            env=self._env_constructor(),
            episodes_per_interaction=self._episodes_per_interaction,
        )
