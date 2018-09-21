from typing import Callable

import gym

from rl_trainer.agent.replay_buffer import ReplayBuffer
from rl_trainer.commons import ExperienceTuple
from rl_trainer.ddpg_impl.flower.actor_critic.critic import Critic
import tensorflow as tf
import numpy as np

from rl_trainer.ddpg_impl.flower.actor_critic import Actor


class Agent:
    def __init__(self, actor: Actor, critic: Critic,
                 replay_buffer: ReplayBuffer, actor_noise: Callable):
        self.actor_noise = actor_noise
        self.replay_buffer = replay_buffer
        self.critic = critic
        self.actor = actor

    def act(self, current_state):
        action = self.actor.predict(
            states_batch=np.array([current_state + self.actor_noise()]),
        )[0]  # unpack actions batch of size 1
        return action


class Train:
    def __init__(self, sess: tf.Session, env: gym.Env, actor: Actor,
                 critic: Critic, actor_noise: Callable, replay_buffer: ReplayBuffer,
                 tf_summary_dir: str, agent: Agent):
        self._agent = agent
        self._tf_summary_dir = tf_summary_dir
        self._actor_noise = actor_noise
        self._critic = critic
        self._actor = actor
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
        self._agent.actor.update_target_network()
        self._agent.critic.update_target_network()

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

                self._agent.replay_buffer.add(ExperienceTuple(
                    initial_state=current_state,
                    action=action,
                    reward=reward,
                    final_state=new_state,
                    final_state_is_terminal=done,
                ))

                if self._agent.replay_buffer.can_provide_samples():
                    self._train_with_replay_buffer(batch_size)

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

    def _train_with_replay_buffer(self, batch_size: int):
        batch = self._replay_buffer.sample_batch(batch_size)

        # Calculate targets
        target_q_values = self._critic.predict_target(
            states_batch=np.array(batch.final_states),
            actions_batch=self._actor.predict_target(batch.final_states),
        )

        q_values = []
        for target_q_value, exp_tuple in zip(target_q_values, batch.experience_tuples):
            if exp_tuple.final_state_is_terminal:
                q_values.append(exp_tuple.reward)
            else:
                q_values.append(exp_tuple.reward + self._critic.gamma*target_q_value)

        # Update the critic given the targets
        predicted_q_value, _ = self._critic.train(
            states_batch=np.array(batch.initial_states),
            actions_batch=np.array(batch.actions),
            q_values_batch=np.array(q_values).reshape((-1, 1)),
        )
        self._episode_max_q = np.amax(predicted_q_value)

        self._train_actor(batch.initial_states)
        self._actor.update_target_network()
        self._critic.update_target_network()

    def _train_actor(self, states_batch) -> None:
        # Update the actor policy using the sampled gradient
        actions_batch = self._actor.predict(states_batch=np.array(states_batch))
        action_grads_batch = self._critic.action_gradients(
            states_batch=np.array(states_batch), actions_batch=actions_batch)
        self._actor.train(
            states_batch=np.array(states_batch), action_grads_batch=action_grads_batch[0])

    def build_summaries(self):
        self.episode_reward_ph = tf.placeholder(tf.float32)
        tf.summary.scalar("Reward", self.episode_reward_ph)
        self.episode_max_q_ph = tf.placeholder(tf.float32)
        tf.summary.scalar("Qmax Value", self.episode_max_q_ph)
        self.summary_ops = tf.summary.merge_all()
