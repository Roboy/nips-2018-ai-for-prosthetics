from rl_trainer.agent import RandomAgent
from rl_trainer.commons import MockEnvironment
from rl_trainer.experiment import Experiment


def test_experiment_run():
    agent = RandomAgent(action_space=MockEnvironment.action_space,
                        state_space=MockEnvironment.observation_space)
    Experiment(agent=agent, env=MockEnvironment()).run()
