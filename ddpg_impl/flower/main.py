import gym
from gym import wrappers
import pprint as pp
import tensorflow as tf
import numpy as np
from osim.env import ProstheticsEnv

from flower.action_noise import OrnsteinUhlenbeckActionNoise
from flower.args_parser import setup_args_parser
from flower.ddpg import CriticNetwork
from flower.actor_critic import Actor
from flower.train import train


def main(args):

    with tf.Session() as sess:

        env = gym.make(args["env"])
        #env = ProstheticsEnv(visualize=True)
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

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        if args['use_gym_monitor']:
            if not args['render_env']:
                env = wrappers.Monitor(
                    env, args['monitor_dir'], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args['monitor_dir'], force=True)

        train(sess, env, args, actor, critic, actor_noise)

        if args['use_gym_monitor']:
            env.monitor.close()


if __name__ == '__main__':
    parser = setup_args_parser()
    args = vars(parser.parse_args())
    pp.pprint(args)
    main(args)
