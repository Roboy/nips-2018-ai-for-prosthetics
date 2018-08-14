import argparse
import os

from agent import RandomAgent
from agent_group.agent_group import AgentGroup
from agent_group.parallelizer import MockParallelizer
from serializer import CSVEpisodeSerializer

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("process_number", type=int)
parser.add_argument("num_episodes", type=int)
args = parser.parse_args()

process_number = args.process_number
num_episodes = args.num_episodes

agent_group = AgentGroup(
    episodes_per_step=num_episodes,
    parallelizer=MockParallelizer(num_processes=1),
    initial_agent=RandomAgent(),
)
agent_group.rollout_and_learn()
print(f"Process {process_number} completed its rollouts.")

results_dir = os.path.join("results_dir", "process_{}".format(process_number))
os.makedirs(results_dir)
for idx, episode in enumerate(agent_group.episodes_history):
    episode_fname = os.path.join(results_dir, "episode_{}".format(idx))
    CSVEpisodeSerializer().serialize(episode, out_fname=episode_fname)
print(f"Process {process_number} completed dumped to disk.")

