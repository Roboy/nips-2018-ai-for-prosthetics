from typing import List, Tuple

import gym

from rl_trainer.agent import GymAgent
from rl_trainer.commons import ExperienceTuple, Episode


class Experiment:
    """
    Makes an agent interact with the environemnt and records the
    results into a list of episodes.
    """

    def __init__(self, agent: GymAgent, env: gym.Env, num_episodes: int = 1):
        self._num_episodes = num_episodes
        self._env = env
        self._agent = agent

    def run(self) -> List[Episode]:
        return [self._run_episode() for _ in range(self._num_episodes)]

    def _run_episode(self) -> Episode:
        self._experience_tuples = []
        self._current_state = self._env.reset()
        self._done = False
        while not self._done:
            action = self._agent.act(self._current_state)
            step = self._env.step(action)
            self._process_step(step=step, action=action)
        return Episode(experience_tuples=self._experience_tuples)

    def _process_step(self, step: tuple, action):
        final_state, reward, self._done, info = step
        exp_tuple = ExperienceTuple(
            initial_state=self._current_state,
            action=action,
            reward=reward,
            final_state=final_state,
            final_state_is_terminal=self._done,
        )
        self._experience_tuples.append(exp_tuple)
        self._current_state = final_state
