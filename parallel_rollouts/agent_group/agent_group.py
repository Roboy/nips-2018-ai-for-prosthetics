from typing import List

from agent import ProstheticsEnvAgent
from commons import Episode
from agent_group.parallelizer import Parallelizer
from rollout import RollOutConfiguration


class AgentGroup:
    def __init__(
            self,
            episodes_per_step: int,
            parallelizer: Parallelizer,
            initial_agent: ProstheticsEnvAgent,
    ):
        self._episodes_per_step = episodes_per_step
        self.current_agent = initial_agent
        self._parallelizer = parallelizer
        self.episodes_history: List[Episode] = []

    def rollout_and_learn(self):
        configuration = RollOutConfiguration(
            agent=self.current_agent,
            num_episodes=self._episodes_per_step
        )
        episodes: List[Episode] = self._parallelizer.launch_in_parallel(configuration)
        self.episodes_history.extend(episodes)
        self.current_agent.train(episodes)
