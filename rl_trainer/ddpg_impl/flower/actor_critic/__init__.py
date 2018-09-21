"""
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
from .actor import Actor
from .critic import Critic
from .tf_ddpg_agent import TFDDPGAgent
