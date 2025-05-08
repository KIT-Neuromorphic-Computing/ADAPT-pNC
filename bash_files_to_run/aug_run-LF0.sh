#!/usr/bin/env bash

#SBATCH --job-name=aug-LF-DS0
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=10
#SBATCH --time=00:01:00
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.o%j.%N

chmod u+x aug_run_LearnableFilter_ds0.sh
./aug_run_LearnableFilter_ds0.sh
