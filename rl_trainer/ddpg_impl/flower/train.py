from typing import Callable

from rl_trainer.ddpg_impl.flower.actor_critic.critic import Critic
import tensorflow as tf
import numpy as np

from rl_trainer.ddpg_impl.flower.actor_critic import Actor
from rl_trainer.ddpg_impl.flower.replay_buffer import ReplayBuffer


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_ops = tf.summary.merge_all()

    return summary_ops, episode_reward, episode_ave_max_q


def train(sess: tf.Session, env, args, actor: Actor, critic: Critic, actor_noise: Callable, replay_buffer: ReplayBuffer):
    # Set up summary Ops
    summary_ops, episode_reward, episode_ave_max_q = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Needed to enable BatchNorm.
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)

    for episode_idx in range(int(args['max_episodes'])):

        current_state = env.reset()

        episode_cumulated_reward = 0
        cumulated_max_q = 0

        for step_idx in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()

            action = actor.predict(np.reshape(current_state, (1, actor.s_dim))) + actor_noise()

            new_state, reward, done, _ = env.step(action[0])

            replay_buffer.add(
                np.reshape(current_state, (actor.s_dim,)),
                np.reshape(action, (actor.a_dim,)),
                reward,
                done,
                np.reshape(new_state, (actor.s_dim,)),
            )

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                cumulated_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            current_state = new_state
            episode_cumulated_reward += reward

            if done:
                summary_str = sess.run(summary_ops, feed_dict={
                    episode_reward: episode_cumulated_reward,
                    episode_ave_max_q: cumulated_max_q / float(step_idx)
                })

                writer.add_summary(summary_str, episode_idx)
                writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(
                    int(episode_cumulated_reward), episode_idx, (cumulated_max_q / float(step_idx)))
                )
                break
