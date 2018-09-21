import gym

from rl_trainer.commons import ExperienceTuple
import tensorflow as tf

from rl_trainer.ddpg_impl.flower.actor_critic import TFDDPGAgent


class Train:
    def __init__(self, sess: tf.Session, env: gym.Env, tf_summary_dir: str, agent: TFDDPGAgent):
        self._agent = agent
        self._tf_summary_dir = tf_summary_dir
        self._env = env
        self._sess = sess
        self._episode_reward = 0
        self._episode_max_q = 0

    def train(self, num_episodes: int, max_episode_len: int, render_env: bool):
        self._sess.run(tf.global_variables_initializer())

        # Initialize target network weight
        self._agent.update_target_nets()

        # Needed to enable BatchNorm.
        # This hurts the performance on Pendulum but could be useful
        # in other environments.
        # tflearn.is_training(True)

        for episode_idx in range(num_episodes):

            self._episode_reward = 0
            self._episode_max_q = 0

            current_state = self._env.reset()

            for step_idx in range(max_episode_len):

                if render_env:
                    self._env.render()
                action = self._agent.act(current_state)
                new_state, reward, done, _ = self._env.step(action)

                self._agent._replay_buffer.extend([
                    ExperienceTuple(
                        initial_state=current_state,
                        action=action,
                        reward=reward,
                        final_state=new_state,
                        final_state_is_terminal=done,
                    ),
                ])

                current_state = new_state
                self._episode_reward += reward

                if done:
                    self.log_episode(episode_idx)
                    break

    def log_episode(self, episode_idx):
        print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(
            int(self._episode_reward),
            episode_idx,
            self._agent.episode_max_q,  # the trainer shouldnt know anything about any q_values
        ))
