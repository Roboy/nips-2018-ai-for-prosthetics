# nips-2018-ai-for-prosthetics

## multiprocess\_envs.ipynb
It contains code for running the env in parallel. Please install OpenAI baselines
from (here)[https://github.com/openai/baselines.git]. You also need pytorch.

The nproc is number of processes you want to run in parallel (equal to number of cores)
The network random\_policy is a 3 layer network that takes in observations and outputs actions. (Feel free to modify it)
The batch size is equal to the nproc.
The output is dictionary _dataset_. The data is stored as list of numpy arrays [next\_obs,action,obs]
