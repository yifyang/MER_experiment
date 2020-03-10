#!/usr/bin/env bash

source activate mer
seed=1
PERM="--n_layers 2 --n_hiddens 100 --data_path data --save_path results --batch_size 1 --log_every 100 --samples_per_task 1000 --data_file mnist_permutations.pt --cuda no  --seed"
python main.py $PERM $seed --model gem --lr 0.01 --n_memories 25 --memory_strength 1.0
