import os
import shutil

import gym

from rl_trainer.ddpg_impl.flower.args_parser import RESULTS_DIR
from rl_trainer.ddpg_impl.flower.args_parser import setup_args_parser
from rl_trainer.ddpg_impl.flower.main import main


def test_main():
    """
    MockEnvironment didnt reduce test time. Probably the fact of importing
    openAI gym + TensorFlow makes this test slow.
    """
    try:
        parser = setup_args_parser()
        args = vars(parser.parse_args())
        args["max_episodes"] = 2
        main(args, env=gym.make(args["env"]))
    finally:
        if os.path.exists(RESULTS_DIR):
            shutil.rmtree(RESULTS_DIR)
        os.remove(os.path.join(os.path.dirname(RESULTS_DIR), "err.log"))
        os.remove(os.path.join(os.path.dirname(RESULTS_DIR), "out.log"))
