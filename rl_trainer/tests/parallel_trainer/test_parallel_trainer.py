from rl_trainer.agent.gym_agent import MockSpace
from rl_trainer.agent import RandomAgent
from rl_trainer.parallel_trainer.parallel_trainer import ParallelTrainer


class MockEnvironment:
    def reset(self):
        pass

    def step(self, action):
        return 1, 2, 3, 4


def test_construction():
    ParallelTrainer(
        initial_agent=None,
        env_constructor=lambda: None,
        episodes_per_interaction=1,
        num_processes=1,
    )


def test_training_step():
    space = MockSpace(2)
    ParallelTrainer(RandomAgent(space, space), lambda: MockEnvironment(), 1, 1).training_step()
