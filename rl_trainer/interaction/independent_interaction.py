from typing import List

import gym

from rl_trainer.agent import GymAgent
from rl_trainer.commons import ExperienceTuple, Episode


class IndependentInteraction:
    """
    The independent interaction is the atomic parallelization unit. It
    does not receive any information about other independent
    interactions that may occur in parallel.
    """

    def __init__(self, agent: GymAgent, env: gym.Env, num_episodes: int = 1):
        self._num_episodes = num_episodes
        self._env = env
        self._agent = agent

    def run(self) -> List[Episode]:
        return [self._run_episode() for _ in range(self._num_episodes)]

    def _run_episode(self) -> Episode:
        experience_tuples: List[ExperienceTuple] = []
        initial_state = self._env.reset()
        done = False
        while not done:
            action = self._agent.act(initial_state)
            final_state, reward, done, info = self._env.step(action)
            experience_tuple = ExperienceTuple(
                initial_state=initial_state,
                action=action,
                reward=reward,
                final_state=final_state,
                final_state_is_terminal=done
            )
            experience_tuples.append(experience_tuple)
            initial_state = final_state
        return Episode(experience_tuples=experience_tuples)
