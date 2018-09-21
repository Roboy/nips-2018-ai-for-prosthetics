import gym

from rl_trainer.agent.replay_buffer import ReplayBuffer
from rl_trainer.commons import ExperienceTuple
import tensorflow as tf

from rl_trainer.ddpg_impl.flower.actor_critic import TFDDPGAgent


class Train:
    def __init__(self, sess: tf.Session, env: gym.Env, replay_buffer: ReplayBuffer,
                 tf_summary_dir: str, agent: TFDDPGAgent):
        self._agent = agent
        self._tf_summary_dir = tf_summary_dir
        self._env = env
        self._replay_buffer = replay_buffer
        self._sess = sess
        self._episode_reward = 0
        self._episode_max_q = 0

    def train(self, num_episodes: int, max_episode_len: int, batch_size: int, render_env: bool):
        self.build_summaries()
        self._sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(self._tf_summary_dir, self._sess.graph)

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

                if self._agent._replay_buffer.can_provide_samples():
                    self._agent._train_with_replay_buffer(batch_size)

                current_state = new_state
                self._episode_reward += reward

                if done:
                    self._episode_max_q = self._agent.episode_max_q  # the trainer shouldnt know anything about any q_values
                    self.log_episode(episode_idx, step_idx)
                    break

    def log_episode(self, episode_idx, step_idx):
        summary_str = self._sess.run(self.summary_ops, feed_dict={
            self.episode_reward_ph: self._episode_reward,
            self.episode_max_q_ph: self._episode_max_q,
        })
        self.writer.add_summary(summary_str, episode_idx)
        self.writer.flush()
        print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(
            int(self._episode_reward),
            episode_idx,
            (self._episode_max_q / float(step_idx))
        ))



    def build_summaries(self):
        self.episode_reward_ph = tf.placeholder(tf.float32)
        tf.summary.scalar("Reward", self.episode_reward_ph)
        self.episode_max_q_ph = tf.placeholder(tf.float32)
        tf.summary.scalar("Qmax Value", self.episode_max_q_ph)
        self.summary_ops = tf.summary.merge_all()
