import argparse
import os

from agent import RandomAgent
from agent import OneMuscleAgent
from rollout import RollOut

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("process_number", type=int)
parser.add_argument("num_episodes", type=int)
parser.add_argument("worker_type", type=int)
args = parser.parse_args()

process_number = args.process_number
num_episodes = args.num_episodes
results_dir = os.path.join("results_dir", "process_{}".format(process_number))

if args.worker_type == 0:
    rollout = RollOut(
        agent=OneMuscleAgent(),
        visualize=False,
        output_dir=results_dir+"OneMuscle",
    )
    print("OneMuscle")
else:
    rollout = RollOut(
        agent=RandomAgent(),
        visualize=False,
        output_dir=results_dir+"Random",
    )
    print("Random")
rollout.start(num_episodes)
