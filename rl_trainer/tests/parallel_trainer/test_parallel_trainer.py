from rl_trainer.commons import MockEnvironment
from rl_trainer.agent import RandomAgent
from rl_trainer.parallel_trainer.parallel_trainer import ParallelTrainer


def test_training_step():
    agent = RandomAgent(action_space=MockEnvironment.action_space,
                        state_space=MockEnvironment.state_space)
    ParallelTrainer(agent, lambda: MockEnvironment(), 1, 1).training_step()
