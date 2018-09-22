from rl_trainer.commons import MockEnvironment
from rl_trainer.agent import RandomAgent
from rl_trainer.parallelizer import ExperimentConfig, LocalParallelizer


def test_run_experiments():
    parallelizer = LocalParallelizer(num_processes=1)
    config = ExperimentConfig(
        agent=RandomAgent(action_space=MockEnvironment.action_space),
        environment_constructor=MockEnvironment,
        episodes_per_experiment=3,
    )
    parallelizer.run_experiments_with_config(config)
