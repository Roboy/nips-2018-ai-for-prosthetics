# Run Parallel Rollouts
## Installation
Setup the Python environment according to: https://github.com/stanfordnmbl/osim-rl
## Running
Where NUM_PROCESSES is the number of parallel processes and NUM_EPISODES is the 
number of episodes each process should run for. 
````bash
bash launch.sh NUM_PROCESSES NUM_EPISODES
````
The output will be saved in ```results_dir``` and any errors will be logged in 
the ``logs`` file.