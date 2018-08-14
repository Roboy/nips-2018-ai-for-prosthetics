import argparse
import os

from agent import RandomAgent
from rollout import RollOut
from serializer import CSVEpisodeSerializer

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("process_number", type=int)
parser.add_argument("num_episodes", type=int)
args = parser.parse_args()

process_number = args.process_number
num_episodes = args.num_episodes

episodes = RollOut(agent=RandomAgent()).get_episodes(num_episodes)
print(f"Process {process_number} completed its rollouts.")

results_dir = os.path.join("results_dir", "process_{}".format(process_number))
os.makedirs(results_dir)
for idx, episode in enumerate(episodes):
    episode_fname = os.path.join(results_dir, "episode_{}".format(idx))
    CSVEpisodeSerializer().serialize(episode, out_fname=episode_fname)
print(f"Process {process_number} completed dumped to disk.")

