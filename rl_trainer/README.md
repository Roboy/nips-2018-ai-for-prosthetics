# Run Parallel Rollouts
## Installation
Setup the Python environment according to: https://github.com/stanfordnmbl/osim-rl
## Experimentation
Use the ``demo.py`` script to experiment with your agent.
```bash
python -m rl_trainer.demo
```
An example of multi-core experiments is in the ``demo_parallel.py`` script.
```bash
python -m rl_trainer.demo_parallel
```
## Test
```bash
pytest
```