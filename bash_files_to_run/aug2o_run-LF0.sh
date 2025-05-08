#!/usr/bin/env bash

#SBATCH --job-name=0-aug2o_LF-DS
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=10
#SBATCH --time=08:00:00
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.o%j.%N

cp -r ./utils $TMPDIR/

conda shell.bash activate train-model
chmod u+x aug2o_run_LearnableFilter_ds0.sh
./aug2o_run_LearnableFilter_ds0.sh
