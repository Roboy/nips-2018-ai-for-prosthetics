import os

from osim.env import ProstheticsEnv

from rl_trainer.agent import RandomAgent
from rl_trainer.parallel_trainer import ParallelTrainer

from rl_trainer.serializer import CSVEpisodeSerializer

if __name__ == '__main__':
    num_parallel_processes = 2
    learning_iterations = 2
    episodes_per_interaction = 2
    results_dir = "results_dir"
    env_constructor = lambda: ProstheticsEnv(visualize=False)

    assert not os.path.isdir(results_dir), f"Folder '{results_dir}' already exists"

    env = env_constructor()
    agent = RandomAgent(action_space=env.action_space, state_space=env.observation_space)
    trainer = ParallelTrainer(
        env_constructor=env_constructor,
        initial_agent=agent,
        episodes_per_interaction=episodes_per_interaction,
        num_processes=num_parallel_processes,
    )

    for _ in range(learning_iterations):
        trainer.training_step()
    print(f"All learning iterations of {trainer} complete")

    os.makedirs(results_dir)
    for idx, episode in enumerate(trainer.episodes_history):
        episode_fname = os.path.join(results_dir, f"episode_{idx}")
        CSVEpisodeSerializer().serialize(episode, out_fname=episode_fname)
    print("Dump to disk complete")
