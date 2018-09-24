import gym
import numpy as np
from typing import Callable, Collection

import tensorflow as tf
import tflearn
from typeguard import typechecked

from rl_trainer.agent.replay_buffer import ReplayBuffer, InMemoryReplayBuffer
from rl_trainer.commons import Episode, ExperienceTupleBatch
from rl_trainer.ddpg_impl.flower.action_noise import OrnsteinUhlenbeckActionNoise
from . import Actor, Critic


class TFDDPGAgent:

    @typechecked
    def __init__(self, sess: tf.Session, state_dim: int, action_space: gym.spaces.Box,
                 gamma: float = 0.99, replay_buffer: ReplayBuffer = None,
                 actor_noise: Callable = None, actor: Actor = None, critic: Critic = None):

        action_dim = action_space.shape[0]
        action_bound = action_space.high
        self._gamma = gamma

        sess = sess if sess else tf.Session()
        self._critic = critic if critic else Critic(
            sess=sess,
            state_dim=state_dim,
            action_dim=action_dim
        )
        self._actor = actor if actor else Actor(
            sess=sess,
            state_dim=state_dim,
            action_dim=action_dim,
            action_bound=action_bound
        )
        sess.run(tf.global_variables_initializer())

        self._actor_noise = actor_noise if actor_noise else OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(action_dim))
        self._replay_buffer = replay_buffer if replay_buffer else InMemoryReplayBuffer()
        self.episode_max_q = 0

        self._update_target_nets()

    @typechecked
    def act(self, current_state: Collection[float]):
        if self._replay_buffer.has_sufficient_samples():
            self._train()
        tflearn.is_training(False)
        action = self._actor._online_nn.predict(states_batch=np.array([current_state]))
        return action[0] + self._actor_noise()  # unpack tf batch shape

    def _update_target_nets(self):
        self._actor.target_nn_update()
        self._critic.target_nn_update()

    def _train(self):
        tflearn.is_training(True)
        batch = self._replay_buffer.sample_batch()
        self._train_critic(batch)
        self._train_actor(batch)
        self._update_target_nets()

    def _train_critic(self, batch: ExperienceTupleBatch) -> None:
        target_q_values = self._critic.target_nn.predict(
            states_batch=np.array(batch.final_states),
            actions_batch=self._actor.target_nn_predict(states_batch=batch.final_states),
        )

        q_values = []
        for target_q_value, exp_tuple in zip(target_q_values, batch.experience_tuples):
            done = exp_tuple.state_2_is_terminal
            q_values.append(exp_tuple.reward + (1-done)*self._gamma*target_q_value)

        self._critic.online_nn_train(
            states_batch=np.array(batch.initial_states),
            actions_batch=np.array(batch.actions),
            q_values_batch=np.array(q_values).reshape((-1, 1)),
        )

        self._log_max_q(batch=batch)

    def _train_actor(self, batch: ExperienceTupleBatch) -> None:
        """Update the actor policy using the sampled gradient"""
        initial_states = np.array(batch.initial_states)
        actions_batch = self._actor.online_nn_predict(states_batch=initial_states)
        action_grads_batch = self._critic.online_nn_action_gradients(states_batch=initial_states,
                                                                     actions_batch=actions_batch)
        self._actor.online_nn_train(states_batch=initial_states,
                                    action_grads_batch=action_grads_batch[0])

    @typechecked
    def _log_max_q(self, batch: ExperienceTupleBatch):
        predicted_q_values = self._critic.online_nn.predict(
            states_batch=batch.initial_states, actions_batch=batch.actions)
        self.episode_max_q = np.amax(predicted_q_values)

    @typechecked
    def observe_episode(self, episode: Episode):
        self._replay_buffer.extend(episode.experience_tuples)
