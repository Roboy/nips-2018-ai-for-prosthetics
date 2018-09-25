from unittest.mock import MagicMock

from rl_trainer.agent import RandomAgent
from rl_trainer.commons import MockEnvironment
from rl_trainer.experiment import Experiment


def test_experiment_run():
    agent = RandomAgent(action_space=MockEnvironment.action_space)
    Experiment(agent=agent, env=MockEnvironment()).run()


def test_render_env_is_called():
    agent = RandomAgent(action_space=MockEnvironment.action_space)
    env = MockEnvironment()
    env.render = MagicMock()
    Experiment(agent=agent, env=env, render_env=True).run()

    env.render.assert_called()
