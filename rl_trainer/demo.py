from gym.envs.classic_control import PendulumEnv

from rl_trainer.agent import RandomAgent
from rl_trainer.experiment import Experiment

env = PendulumEnv()
experiment = Experiment(
    agent=RandomAgent(action_space=env.action_space),
    env=env,
    num_episodes=5,
)

experiment.run(seed=0)
