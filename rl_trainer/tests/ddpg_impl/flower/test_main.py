import os
import shutil

import pytest

from rl_trainer.commons import MockEnvironment
from rl_trainer.ddpg_impl.flower.args_parser import RESULTS_DIR
from rl_trainer.ddpg_impl.flower.main import main


@pytest.mark.integration
def test_main():
    """
    MockEnvironment didnt reduce test time. Probably the fact of importing
    openAI gym + TensorFlow makes this test slow.
    """
    try:
        main(
            max_episodes=2,
            max_episode_len=100,
            env=MockEnvironment(episode_length=100)
        )
    finally:
        if os.path.exists(RESULTS_DIR):
            shutil.rmtree(RESULTS_DIR)
        os.remove(os.path.join(os.path.dirname(RESULTS_DIR), "err.log"))
        os.remove(os.path.join(os.path.dirname(RESULTS_DIR), "out.log"))
