#!/usr/bin/env bash

#SBATCH --job-name=RNN-baseline
#SBATCH --partition=single
#SBATCH --ntasks-per-node=10
#SBATCH --time=72:00:00
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.o%j.%N

chmod u+x run_baseline_RNN_python.sh
./run_baseline_RNN_python.sh
