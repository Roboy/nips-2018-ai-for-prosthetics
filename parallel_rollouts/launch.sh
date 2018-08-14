#!/usr/bin/env bash

NUM_PROCESSES=$1
NUM_EPISODES=$2
for ((process=1; process<=NUM_PROCESSES; process++)); do
   python orchestrator.py ${process} ${NUM_EPISODES} >>logs 2>>logs &
done