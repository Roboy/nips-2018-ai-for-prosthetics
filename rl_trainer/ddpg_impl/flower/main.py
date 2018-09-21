import gym
from gym import wrappers
import pprint as pp
import tensorflow as tf
import numpy as np
from osim.env import ProstheticsEnv

from rl_trainer.agent.replay_buffer import InMemoryReplayBuffer
from rl_trainer.ddpg_impl.flower.action_noise import OrnsteinUhlenbeckActionNoise
from rl_trainer.ddpg_impl.flower.args_parser import setup_args_parser
from rl_trainer.ddpg_impl.flower.actor_critic import Actor, Critic
from rl_trainer.ddpg_impl.flower.train import Train, TFDDPGAgent


def main(args, env: gym.Env):

    with tf.Session() as sess:
        seed = int(args['random_seed'])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        env.seed(seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        # assert (env.action_space.high == -env.action_space.low)

        actor = Actor(sess, state_dim, action_dim, action_bound,
                      float(args['actor_lr']), float(args['tau']),
                      int(args['minibatch_size']))

        critic = Critic(sess, state_dim, action_dim,
                        float(args['critic_lr']), float(args['tau']),
                        float(args['gamma']),
                        actor.num_trainable_vars)

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        replay_buffer = InMemoryReplayBuffer(buffer_size=int(args["buffer_size"]),
                                             lower_size_limit=int(args["minibatch_size"]),
                                             seed=int(args["random_seed"]))

        if args["use_gym_monitor"]:
            env = wrappers.Monitor(env, args['monitor_dir'], force=True)

        train = Train(
            sess=sess,
            env=env,
            tf_summary_dir=args['summary_dir'],
            agent=TFDDPGAgent(
                actor=actor,
                critic=critic,
                replay_buffer=replay_buffer,
                actor_noise=actor_noise,
            )
        )
        train.train(
            num_episodes=int(args['max_episodes']),
            max_episode_len=int(args["max_episode_len"]),
            render_env=args['render_env'],
        )

        if args['use_gym_monitor']:
            env.monitor.close()


if __name__ == '__main__':
    parser = setup_args_parser()
    args = vars(parser.parse_args())
    pp.pprint(args)
    env = gym.make(args["env"])
    #env = ProstheticsEnv(visualize=True)
    main(args, env)
