# nips-2018-ai-for-prosthetics

## multiprocess\_envs.ipynb
It contains code for running the env in parallel. Please install OpenAI baselines
from (here)[https://github.com/openai/baselines.git]. You also need pytorch.

The nproc is number of processes you want to run in parallel (equal to number of cores)
The network random\_policy is a 3 layer network that takes in observations and outputs actions. (Feel free to modify it)
The batch size is equal to the nproc.
The output is dictionary _dataset_. The data is stored as list of numpy arrays [next\_obs,action,obs]


## For using the DDPG, checkout to the ddpg branch.

### How to setup environment?
sh setup_conda.sh

source activate opensim-rl

sh setup_env.sh

Congrats! Now you are ready to check our agents.

### Run DDPG agent
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python ddpg/train.py \
    --logdir ./logs_ddpg \
    --num-threads 4 \
    --ddpg-wrapper \
    --skip-frames 5 \
    --fail-reward -0.2 \
    --reward-scale 10 \
    --flip-state-action \
    --actor-layers 64-64 --actor-layer-norm --actor-parameters-noise \
    --actor-lr 0.001 --actor-lr-end 0.00001 \
    --critic-layers 64-32 --critic-layer-norm \
    --critic-lr 0.002 --critic-lr-end 0.00001 \
    --initial-epsilon 0.5 --final-epsilon 0.001 \
    --tau 0.0001

### Evaluate DDPG agent
CUDA_VISIBLE_DEVICES="" PYTHONPATH=./ python ddpg/submit.py \
    --restore-actor-from ./logs_ddpg/actor_state_dict.pkl \
    --restore-critic-from ./logs_ddpg/critic_state_dict.pkl \
    --restore-args-from ./logs_ddpg/args.json \
    --num-episodes 10

