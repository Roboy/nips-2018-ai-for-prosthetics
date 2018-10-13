import os
import shutil

import pytest

from rl_trainer.commons import MockEnvironment
from rl_trainer.ddpg_impl.flower.actor_critic.tf_model_saver import TFModelSaver
from rl_trainer.ddpg_impl.flower.main import main

RESULTS_DIR = "./testing_main_results_dir"
GYM_DIR = os.path.join(RESULTS_DIR, "gym/")


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
            env=MockEnvironment(episode_length=100),
            gym_dir=GYM_DIR,
        )
        assert os.path.exists(RESULTS_DIR)
        assert os.path.exists(GYM_DIR)
        assert os.path.exists(TFModelSaver.DEFAULT_MODEL_DIR)
    finally:
        if os.path.exists(RESULTS_DIR):
            shutil.rmtree(RESULTS_DIR)
        if os.path.exists(TFModelSaver.DEFAULT_MODEL_DIR):
            shutil.rmtree(TFModelSaver.DEFAULT_MODEL_DIR)
        try:
            os.remove(os.path.join(os.path.dirname(RESULTS_DIR), "err.log"))
            os.remove(os.path.join(os.path.dirname(RESULTS_DIR), "out.log"))
        except FileNotFoundError:
            pass
