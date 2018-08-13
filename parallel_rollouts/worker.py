import argparse
import os

from agent import RandomAgent
from rollout import RollOut

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("process_number", type=int)
parser.add_argument("num_episodes", type=int)
args = parser.parse_args()

process_number = args.process_number
num_episodes = args.num_episodes
results_dir = os.path.join("results_dir", "process_{}".format(process_number))

rollout = RollOut(
    agent=RandomAgent(),
    visualize=False,
    output_dir=results_dir,
)
rollout.start(num_episodes)
