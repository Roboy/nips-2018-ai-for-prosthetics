import gym

from rl_trainer.commons import ExperienceTuple, Episode

from rl_trainer.ddpg_impl.flower.actor_critic import TFDDPGAgent


class Train:
    def __init__(self, env: gym.Env, agent: TFDDPGAgent):
        self._agent = agent
        self._env = env

    def train(self, num_episodes: int, max_episode_len: int, render_env: bool):
        # Needed to enable BatchNorm.
        # This hurts the performance on Pendulum but could be useful
        # in other environments.

        for episode_idx in range(num_episodes):

            experience_tuples = []
            episode_reward = 0

            current_state = self._env.reset()

            for step_idx in range(max_episode_len):

                if render_env:
                    self._env.render()
                action = self._agent.act(current_state)
                new_state, reward, done, _ = self._env.step(action)

                exp_tup = ExperienceTuple(initial_state=current_state, action=action,
                                          reward=reward, final_state=new_state,
                                          final_state_is_terminal=done)
                experience_tuples.append(exp_tup)

                current_state = new_state
                episode_reward += reward

                if done:
                    self._agent.observe_episode(Episode(experience_tuples))
                    self.log_episode(episode_idx, episode_reward)
                    break

    def log_episode(self, episode_idx: int, reward: float):
        print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(
            int(reward),
            episode_idx,
            self._agent.episode_max_q,  # the trainer shouldnt know anything about any q_values
        ))
