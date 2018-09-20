from typing import Callable

import gym

from rl_trainer.ddpg_impl.flower.actor_critic.critic import Critic
import tensorflow as tf
import numpy as np

from rl_trainer.ddpg_impl.flower.actor_critic import Actor
from rl_trainer.ddpg_impl.flower.replay_buffer import ReplayBuffer


class Train:
    def __init__(self, sess: tf.Session, env: gym.Env, actor: Actor,
                 critic: Critic, actor_noise: Callable, replay_buffer: ReplayBuffer, args):
        self._args = args
        self._actor_noise = actor_noise
        self._critic = critic
        self._actor = actor
        self._env = env
        self._replay_buffer = replay_buffer
        self._sess = sess
        self._episode_reward = 0
        self._episode_max_q = 0

    def train(self):
        self.build_summaries()
        self._sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(self._args['summary_dir'], self._sess.graph)

        # Initialize target network weights
        self._actor.update_target_network()
        self._critic.update_target_network()

        # Needed to enable BatchNorm.
        # This hurts the performance on Pendulum but could be useful
        # in other environments.
        # tflearn.is_training(True)

        for episode_idx in range(int(self._args['max_episodes'])):

            self._episode_reward = 0
            self._episode_max_q = 0

            current_state = self._env.reset()

            for step_idx in range(int(self._args['max_episode_len'])):

                if self._args['render_env']:
                    self._env.render()

                action = self._actor.predict(np.reshape(current_state, (1, self._actor.s_dim))) + self._actor_noise()

                new_state, reward, done, _ = self._env.step(action[0])

                self._replay_buffer.add(
                    np.reshape(current_state, (self._actor.s_dim,)),
                    np.reshape(action, (self._actor.a_dim,)),
                    reward,
                    done,
                    np.reshape(new_state, (self._actor.s_dim,)),
                )

                if self._replay_buffer.size() > int(self._args['minibatch_size']):
                    self._train_actor_critic()

                current_state = new_state
                self._episode_reward += reward

                if done:
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

    def _train_actor_critic(self):
        state_batch, action_batch, reward_batch, done_batch, final_state_batch = \
            self._replay_buffer.sample_batch(int(self._args['minibatch_size']))

        # Calculate targets
        target_q = self._critic.predict_target(
            final_state_batch, self._actor.predict_target(final_state_batch))

        y_i = []
        for k in range(int(self._args['minibatch_size'])):
            if done_batch[k]:
                y_i.append(reward_batch[k])
            else:
                y_i.append(reward_batch[k] + self._critic.gamma * target_q[k])

        # Update the critic given the targets
        predicted_q_value, _ = self._critic.train(
            state_batch, action_batch, np.reshape(y_i, (int(self._args['minibatch_size']), 1)))

        self._episode_max_q = np.amax(predicted_q_value)
        self._train_actor(state_batch)
        self._actor.update_target_network()
        self._critic.update_target_network()

    def _train_actor(self, state_batch) -> None:
        # Update the actor policy using the sampled gradient
        a_outs = self._actor.predict(state_batch)
        grads = self._critic.action_gradients(state_batch, a_outs)
        self._actor.train(state_batch, grads[0])

    def build_summaries(self):
        self.episode_reward_ph = tf.placeholder(tf.float32)
        tf.summary.scalar("Reward", self.episode_reward_ph)
        self.episode_max_q_ph = tf.placeholder(tf.float32)
        tf.summary.scalar("Qmax Value", self.episode_max_q_ph)
        self.summary_ops = tf.summary.merge_all()
