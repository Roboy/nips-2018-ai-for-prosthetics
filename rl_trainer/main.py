from rl_trainer.agent import RandomAgent
from rl_trainer.parallelizer import LocalParallelizer, ExperimentConfig
from gym.envs.classic_control import PendulumEnv


ENV_CONSTRUCTOR = PendulumEnv
AGENT = RandomAgent(action_space=ENV_CONSTRUCTOR().action_space)
NUM_CORES = 2
EPISODES = 10

config = ExperimentConfig(
    agent=AGENT,
    environment_constructor=ENV_CONSTRUCTOR,
    episodes_per_experiment=EPISODES,
)

parallelizer = LocalParallelizer(num_processes=NUM_CORES)
parallelizer.run_experiments_with_config(config)
print("finished")
