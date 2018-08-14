from typing import List

from osim.env import ProstheticsEnv

from agent import ProstheticsEnvAgent
from commons import ExperienceTuple, Episode


class RollOut:
    def __init__(self, agent: ProstheticsEnvAgent, visualize: bool = False,
                 seed: int = 0):
        self._agent = agent
        self._env = ProstheticsEnv(visualize=visualize)
        self._env.change_model(model="3D", prosthetic=True, seed=seed)

    def get_episodes(self, num_episodes: int) -> List[Episode]:
        return [self._run_episode() for _ in range(num_episodes)]

    def _run_episode(self) -> Episode:
        experience_tuples: List[ExperienceTuple] = []
        initial_state = self._env.reset()
        action = self._agent.act(initial_state)
        done = False
        while not done:
            final_state, reward, done, info = self._env.step(action)
            experience_tuple = ExperienceTuple(initial_state, action, final_state)
            experience_tuples.append(experience_tuple)
            initial_state = final_state
        return Episode(experience_tuples=experience_tuples)
