import os

from agent import RandomAgent
from agent_group.agent_group import AgentGroup
from agent_group.parallelizer import MultiProcessingParallelizer
from serializer import CSVEpisodeSerializer

if __name__ == '__main__':
    num_parallel_processes = 1
    learning_iterations = 1
    episodes_per_rollout = 1

    agent_group = AgentGroup(
        episodes_per_rollout=episodes_per_rollout,
        parallelizer=MultiProcessingParallelizer(num_processes=num_parallel_processes),
        initial_agent=RandomAgent(),
    )

    for _ in range(learning_iterations):
        agent_group.rollout_and_learn()
    print("All learning iterations of {} complete".format(agent_group))

    results_dir = "results_dir"
    os.makedirs(results_dir)
    for idx, episode in enumerate(agent_group.episodes_history):
        episode_fname = os.path.join(results_dir, "episode_{}".format(idx))
        CSVEpisodeSerializer().serialize(episode, out_fname=episode_fname)
    print("Dump to disk complete")
