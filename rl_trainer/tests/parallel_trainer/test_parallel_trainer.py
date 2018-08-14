from rl_trainer.parallel_trainer.parallel_trainer import ParallelTrainer


def test_construction():
    ParallelTrainer(
        initial_agent=None,
        env_constructor=lambda _: None,
        episodes_per_interaction=1,
        num_processes=1,
    )
