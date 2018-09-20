import shutil

from flower.args_parser import setup_args_parser, RESULTS_DIR
from flower.main import main


def test_main():
    parser = setup_args_parser()
    args = vars(parser.parse_args())
    args["max_episodes"] = 1
    main(args)
    shutil.rmtree(RESULTS_DIR)
