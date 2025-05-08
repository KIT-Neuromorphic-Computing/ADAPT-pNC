#!/usr/bin/env bash

#SBATCH --job-name=aug-LF-DS13
#SBATCH --partition=single
#SBATCH --ntasks-per-node=10
#SBATCH --time=72:00:00
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.o%j.%N

chmod u+x aug_run_LearnableFilter_ds13.sh
./aug_run_LearnableFilter_ds13.sh
