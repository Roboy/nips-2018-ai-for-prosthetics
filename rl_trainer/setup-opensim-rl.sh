#!/usr/bin/env bash
conda create -n opensim-rl -c kidzik opensim python=3.6.1 -y
source activate opensim-rl
conda install -c conda-forge lapack git
pip install git+https://github.com/stanfordnmbl/osim-rl.git