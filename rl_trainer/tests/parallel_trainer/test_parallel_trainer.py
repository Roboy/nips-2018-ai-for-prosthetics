from rl_trainer.commons import MockEnvironment
from rl_trainer.agent import RandomAgent
from rl_trainer.parallel_trainer.parallel_trainer import ParallelTrainer


def test_training_step():
    agent = RandomAgent(action_space=MockEnvironment.action_space,
                        state_space=MockEnvironment.observation_space)
    ParallelTrainer(agent, lambda: MockEnvironment(), episodes_per_experiment=1,
                    num_processes=1).training_step()
