import numpy as np
from typing import Callable

from rl_trainer.agent.replay_buffer import ReplayBuffer
from . import Actor, Critic


class TFDDPGAgent:
    def __init__(self, actor: Actor, critic: Critic,
                 replay_buffer: ReplayBuffer, actor_noise: Callable):
        self._actor_noise = actor_noise
        self._replay_buffer = replay_buffer
        self._critic = critic
        self._actor = actor

    def act(self, current_state):
        self._train()
        action = self._actor.predict(states_batch=np.array([current_state]))
        return action[0] + self._actor_noise()  # unpack tf batch shape

    def _train(self):
        if self._replay_buffer.can_provide_samples():
            self._train_with_replay_buffer()

    def update_target_nets(self):
        self._actor.update_target_network()
        self._critic.update_target_network()

    def _train_with_replay_buffer(self):
        batch = self._replay_buffer.sample_batch()

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
        self.episode_max_q = np.amax(predicted_q_value)

        self._train_actor(batch.initial_states)
        self.update_target_nets()

    def _train_actor(self, states_batch) -> None:
        # Update the actor policy using the sampled gradient
        actions_batch = self._actor.predict(states_batch=np.array(states_batch))
        action_grads_batch = self._critic.action_gradients(
            states_batch=np.array(states_batch), actions_batch=actions_batch)
        self._actor.train(
            states_batch=np.array(states_batch), action_grads_batch=action_grads_batch[0])


