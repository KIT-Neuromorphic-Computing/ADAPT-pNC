#!/usr/bin/env bash

#SBATCH --job-name=1-aug2o_LF-DS
#SBATCH --partition=single
#SBATCH --ntasks-per-node=10
#SBATCH --time=72:00:00
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.o%j.%N


chmod u+x aug2o_run_LearnableFilter_ds1.sh
./aug2o_run_LearnableFilter_ds1.sh
