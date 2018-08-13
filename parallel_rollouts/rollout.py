import os
from typing import List

from osim.env import ProstheticsEnv

from agent import ProstheticsEnvAgent
from commons import ExperienceTuple
from serializer import CSVEpisodeSerializer


class RollOut:
    def __init__(self, agent: ProstheticsEnvAgent, output_dir: str,
                 visualize: bool = False, seed: int = 0):
        self._serializer = CSVEpisodeSerializer()
        self._agent = agent
        self._env = ProstheticsEnv(visualize=visualize)
        self._env.change_model(model="3D", prosthetic=True, seed=seed)
        self._output_dir = output_dir
        os.makedirs(output_dir)

    def start(self, num_episodes: int):
        for idx in range(num_episodes):
            episode = self._run_episode()
            fname = os.path.join(self._output_dir, str("episode_{}".format(idx)))
            self._serializer.serialize(episode, fname)

    def _run_episode(self):
        current_episode: List[ExperienceTuple] = []
        initial_state = self._env.reset()
        action = self._agent.act(initial_state)
        done = False
        while not done:
            final_state, reward, done, info = self._env.step(action)
            experience_tuple = ExperienceTuple(initial_state, action, final_state)
            current_episode.append(experience_tuple)
            initial_state = final_state
        return current_episode
