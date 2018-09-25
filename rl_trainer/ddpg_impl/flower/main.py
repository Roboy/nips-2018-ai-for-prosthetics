import gym
from gym import wrappers
import tensorflow as tf
import numpy as np
from gym.envs.classic_control import PendulumEnv
from osim.env import ProstheticsEnv
from typeguard import typechecked

from rl_trainer.ddpg_impl.flower.actor_critic import TensorFlowDDPGAgent
from rl_trainer.experiment import Experiment


@typechecked
def main(max_episodes: int, max_episode_len, env: gym.Env):

    seed = 0
    np.random.seed(seed)
    tf.set_random_seed(seed)
    env.seed(seed)

    render_env = True
    use_gym_monitor = False
    gym_monitor_dir = "./flower_results"

    agent = TensorFlowDDPGAgent(
        state_dim=env.observation_space.shape[0],
        action_space=env.action_space,
    )

    if use_gym_monitor:
        env = wrappers.Monitor(env, gym_monitor_dir, force=True)

    Experiment(
        agent=agent,
        env=env,
        render_env=render_env,
        num_episodes=max_episodes,
        max_episode_len=max_episode_len,
    ).run(seed=seed)

    if use_gym_monitor:
        env.monitor.close()


if __name__ == '__main__':
    env = PendulumEnv()
    #env = ProstheticsEnv(visualize=True)

    main(max_episodes=10, max_episode_len=int(100), env=env)
