import gym
import numpy as np
from typing import Callable, Collection

import tensorflow as tf
import tflearn
from overrides import overrides
from typeguard import typechecked

from rl_trainer.agent import GymAgent
from rl_trainer.agent.replay_buffer import ReplayBuffer, InMemoryReplayBuffer
from rl_trainer.commons import Episode, ExperienceTupleBatch
from rl_trainer.ddpg_impl.flower.actor_critic.tf_model_saver import TFModelSaver
from .action_noise import OrnsteinUhlenbeckActionNoise
from .critic import OnlineCriticNetwork
from .actor import OnlineActorNetwork


class TensorFlowDDPGAgent(GymAgent):

    def __init__(self, state_dim: int, action_space: gym.spaces.Box, sess: tf.Session = None,
                 gamma: float = 0.99, replay_buffer: ReplayBuffer = None,
                 actor_noise: Callable = None, tau: float = 0.001,
                 critic_nn: OnlineCriticNetwork = None, actor_nn: OnlineActorNetwork = None,
                 tf_model_saver: TFModelSaver = None):

        self._model_saver = tf_model_saver if tf_model_saver else TFModelSaver()
        action_dim = action_space.shape[0]
        self._gamma = gamma

        self._sess = sess if sess else tf.Session()

        self._Q = critic_nn if critic_nn else OnlineCriticNetwork(
            sess=self._sess, state_dim=state_dim, action_dim=action_dim)
        self._Qʹ = self._Q.create_target_network(tau=tau)

        self._μ = actor_nn if actor_nn else OnlineActorNetwork(
            action_bound=action_space.high, sess=self._sess,
            state_dim=state_dim, action_dim=action_dim)
        self._μʹ = self._μ.create_target_network(tau=tau)

        self._sess.run(tf.global_variables_initializer())

        self._actor_noise = actor_noise if actor_noise else OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(action_dim))
        self._replay_buffer = replay_buffer if replay_buffer else InMemoryReplayBuffer()

        self.episode_max_q = 0

        self._update_target_nets()

    def _update_target_nets(self):
        self._μʹ.update()
        self._Qʹ.update()

    @typechecked
    @overrides
    def act(self, current_state: Collection[float]):
        if self._replay_buffer.has_sufficient_samples():
            self._train()
        tflearn.is_training(False, session=self._sess)
        s = np.array([current_state])  # pack single state into tf action batch
        action = self._μ(s=s)
        return action[0] + self._actor_noise()  # unpack tf batch shape

    def _train(self):
        tflearn.is_training(True, session=self._sess)
        batch = self._replay_buffer.sample_batch()
        self._train_critic(batch)
        self._train_actor(batch)
        self._update_target_nets()

    @typechecked
    def _train_critic(self, batch: ExperienceTupleBatch) -> None:
        μʹ = self._μʹ
        γ = self._gamma
        s2 = np.array(batch.states_2)
        dones = batch.states_2_are_terminal

        Qs_s2 = self._Qʹ(s=s2, a=μʹ(s=s2))
        yᵢ = [(r + (1-done)*γ*Q_s2) for r, done, Q_s2 in zip(batch.rewards, dones, Qs_s2)]
        yᵢ = np.array(yᵢ).reshape((-1, 1))

        s = np.array(batch.states_1)
        a = np.array(batch.actions)
        self._Q.train(s=s, a=a, y_i=yᵢ)

        self._log_max_q(batch=batch)

    @typechecked
    def _train_actor(self, batch: ExperienceTupleBatch) -> None:
        """Update the actor policy using the sampled gradient"""
        s = np.array(batch.states_1)
        μ = self._μ
        grads_a = self._Q.grads_a(s=s, a=μ(s))
        # TODO: Understand why the grads_a is being unpacked below
        μ.train(s=s, grads_a=grads_a[0])

    @typechecked
    def _log_max_q(self, batch: ExperienceTupleBatch):
        s, a = batch.states_1, batch.actions
        q_vals = self._Q(s=s, a=a)
        self.episode_max_q = np.amax(q_vals)

    @typechecked
    @overrides
    def observe_episode(self, episode: Episode):
        self._replay_buffer.extend(episode.experience_tuples)
        self._model_saver.step(self._sess)

    @overrides
    def set_seed(self, seed: int):
        tf.set_random_seed(seed)
