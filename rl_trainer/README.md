# Run Parallel Rollouts
## Installation
Setup the Python environment according to: https://github.com/stanfordnmbl/osim-rl
## Running
Use the ``orchestrator.py`` script to interaction and learn with your agent in parallel.
```bash
python -m rl_trainer.orchestrator
```
The output will be saved in ```results_dir``` and any errors will be logged in 
the ``logs`` file.
## Test
```bash
pytest
```