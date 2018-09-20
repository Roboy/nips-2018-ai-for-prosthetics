import os
import shutil

from rl_trainer.ddpg_impl.flower.args_parser import RESULTS_DIR
from rl_trainer.ddpg_impl.flower.args_parser import setup_args_parser
from rl_trainer.ddpg_impl.flower.main import main


def test_main():
    try:
        parser = setup_args_parser()
        args = vars(parser.parse_args())
        args["max_episodes"] = 1
        main(args)
    finally:
        shutil.rmtree(RESULTS_DIR)
        os.remove(os.path.join(os.path.dirname(RESULTS_DIR), "err.log"))
        os.remove(os.path.join(os.path.dirname(RESULTS_DIR), "out.log"))
