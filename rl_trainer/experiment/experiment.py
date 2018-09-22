import uuid

import gym
from typeguard import typechecked

from rl_trainer.agent import GymAgent
from rl_trainer.commons import ExperienceTuple, Episode


class Experiment:
    """
    Makes an agent interact with the environment and records the
    results into a list of episodes.
    """

    @typechecked
    def __init__(self, agent: GymAgent, env: gym.Env, render_env: bool = False,
                 num_episodes: int = 1, max_episode_len=int(1e5)):
        self._id = uuid.uuid4()
        self._num_episodes = num_episodes
        self._max_episode_len = max_episode_len
        self._env = env
        self._agent = agent
        self._render_env = render_env

    @typechecked
    def run(self, seed: int = None):
        if seed:
            self._set_seeds(seed)
        for ep_idx in range(self._num_episodes):
            episode = self._rollout_one_episode()
            self._agent.observe_episode(episode)
            self._log_episode(episode, ep_idx)
        print(f"Experiment {self._id} finished")

    def _rollout_one_episode(self) -> Episode:
        experience_tuples = []
        current_state = self._env.reset()
        for idx in range(self._max_episode_len):
            action = self._agent.act(current_state)
            new_state, reward, done, info = self._env.step(action)
            exp_tuple = ExperienceTuple(
                state_1=current_state,
                action=action,
                reward=reward,
                state_2=new_state,
                state_2_is_terminal=done,
            )
            experience_tuples.append(exp_tuple)
            current_state = new_state
            if done:
                break
        return Episode(experience_tuples=experience_tuples)

    def _set_seeds(self, seed: int):
        self._env.seed(seed)

    def _log_episode(self, episode: Episode, episode_num: int):
        episode_reward = sum(tup.reward for tup in episode.experience_tuples)
        print(f"| Experiment: {self._id} | Reward: {episode_reward} "
              f"| Episode: {episode_num} |")
