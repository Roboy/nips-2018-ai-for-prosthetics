import gym
from gym import wrappers
import pprint as pp
import tensorflow as tf
import numpy as np
from osim.env import ProstheticsEnv

from rl_trainer.agent.replay_buffer import InMemoryReplayBuffer
from rl_trainer.ddpg_impl.flower.action_noise import OrnsteinUhlenbeckActionNoise
from rl_trainer.ddpg_impl.flower.args_parser import setup_args_parser
from rl_trainer.ddpg_impl.flower.actor_critic import Actor, Critic, TFDDPGAgent
from rl_trainer.ddpg_impl.flower.train import Train


def main(args, env: gym.Env):

    with tf.Session() as sess:
        seed = int(args['random_seed'])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        env.seed(seed)

        # Ensure action bound is symmetric
        # assert (env.action_space.high == -env.action_space.low)

        if args["use_gym_monitor"]:
            env = wrappers.Monitor(env, args['monitor_dir'], force=True)

        train = Train(
            env=env,
            agent=TFDDPGAgent(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                sess=sess,
                gamma=float(args['gamma']),
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
