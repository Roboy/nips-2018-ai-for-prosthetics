from typing import List, NamedTuple

from osim.env import ProstheticsEnv

from agent import ProstheticsEnvAgent
from commons import ExperienceTuple, Episode


class RollOutConfiguration(NamedTuple):
    agent: ProstheticsEnvAgent
    visualize_env: bool = False
    num_episodes: int = 1
    env_seed: int = 0


class RollOut:
    def __init__(self, configuration: RollOutConfiguration):
        self._configuration = configuration
        self._env: ProstheticsEnv = None  # Is initialized at run()

    def run(self) -> List[Episode]:
        self._env = ProstheticsEnv(visualize=self._configuration.visualize_env)
        self._env.change_model(seed=self._configuration.env_seed)
        return [self._run_episode() for _ in range(self._configuration.num_episodes)]

    def _run_episode(self) -> Episode:
        experience_tuples: List[ExperienceTuple] = []
        initial_state = self._env.reset()
        action = self._configuration.agent.act(initial_state)
        done = False
        while not done:
            final_state, reward, done, info = self._env.step(action)
            experience_tuple = ExperienceTuple(initial_state, action, final_state)
            experience_tuples.append(experience_tuple)
            initial_state = final_state
        return Episode(experience_tuples=experience_tuples)
