#!/bin/bash
    

#SBATCH --job-name=data_augmentation
#SBATCH --partition=single
#SBATCH --ntasks-per-node=40
#SBATCH --time=72:00:00
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.o%j.%N


python3 mlflow_exp_LearnableFilters.py --N_train 20 --e_train 0.0 --DATASET 0  --SEED 0 --task temporal --loss celoss --opt adamw --LR 0.1 --N_feature 1 --metric temporal_acc --projectname 0_AugmentAnalysis  --DEVICE cpu --PATIENCE 100 --augment True --N_feature 1&
python3 mlflow_exp_LearnableFilters.py --N_train 20 --e_train 0.0 --DATASET 1  --SEED 0 --task temporal --loss celoss --opt adamw --LR 0.1 --N_feature 1 --metric temporal_acc --projectname 1_AugmentAnalysis  --DEVICE cpu --PATIENCE 100 --augment True --N_feature 1&
python3 mlflow_exp_LearnableFilters.py --N_train 20 --e_train 0.0 --DATASET 2  --SEED 0 --task temporal --loss celoss --opt adamw --LR 0.1 --N_feature 1 --metric temporal_acc --projectname 2_AugmentAnalysis  --DEVICE cpu --PATIENCE 100 --augment True --N_feature 1&
python3 mlflow_exp_LearnableFilters.py --N_train 20 --e_train 0.0 --DATASET 3  --SEED 0 --task temporal --loss celoss --opt adamw --LR 0.1 --N_feature 1 --metric temporal_acc --projectname 3_AugmentAnalysis  --DEVICE cpu --PATIENCE 100 --augment True --N_feature 1&
python3 mlflow_exp_LearnableFilters.py --N_train 20 --e_train 0.0 --DATASET 4  --SEED 0 --task temporal --loss celoss --opt adamw --LR 0.1 --N_feature 1 --metric temporal_acc --projectname 4_AugmentAnalysis  --DEVICE cpu --PATIENCE 100 --augment True --N_feature 1&
python3 mlflow_exp_LearnableFilters.py --N_train 20 --e_train 0.0 --DATASET 5  --SEED 0 --task temporal --loss celoss --opt adamw --LR 0.1 --N_feature 1 --metric temporal_acc --projectname 5_AugmentAnalysis  --DEVICE cpu --PATIENCE 100 --augment True --N_feature 1&
python3 mlflow_exp_LearnableFilters.py --N_train 20 --e_train 0.0 --DATASET 6  --SEED 0 --task temporal --loss celoss --opt adamw --LR 0.1 --N_feature 1 --metric temporal_acc --projectname 6_AugmentAnalysis  --DEVICE cpu --PATIENCE 100 --augment True --N_feature 1&
python3 mlflow_exp_LearnableFilters.py --N_train 20 --e_train 0.0 --DATASET 7  --SEED 0 --task temporal --loss celoss --opt adamw --LR 0.1 --N_feature 1 --metric temporal_acc --projectname 7_AugmentAnalysis  --DEVICE cpu --PATIENCE 100 --augment True --N_feature 1&
python3 mlflow_exp_LearnableFilters.py --N_train 20 --e_train 0.0 --DATASET 8  --SEED 0 --task temporal --loss celoss --opt adamw --LR 0.1 --N_feature 1 --metric temporal_acc --projectname 8_AugmentAnalysis  --DEVICE cpu --PATIENCE 100 --augment True --N_feature 1&
python3 mlflow_exp_LearnableFilters.py --N_train 20 --e_train 0.0 --DATASET 9  --SEED 0 --task temporal --loss celoss --opt adamw --LR 0.1 --N_feature 1 --metric temporal_acc --projectname 9_AugmentAnalysis  --DEVICE cpu --PATIENCE 100 --augment True --N_feature 1&
python3 mlflow_exp_LearnableFilters.py --N_train 20 --e_train 0.0 --DATASET 10 --SEED 0 --task temporal --loss celoss --opt adamw --LR 0.1 --N_feature 1 --metric temporal_acc --projectname 10_AugmentAnalysis --DEVICE cpu --PATIENCE 100 --augment True --N_feature 1&
python3 mlflow_exp_LearnableFilters.py --N_train 20 --e_train 0.0 --DATASET 11 --SEED 0 --task temporal --loss celoss --opt adamw --LR 0.1 --N_feature 1 --metric temporal_acc --projectname 11_AugmentAnalysis --DEVICE cpu --PATIENCE 100 --augment True --N_feature 1&
python3 mlflow_exp_LearnableFilters.py --N_train 20 --e_train 0.0 --DATASET 12 --SEED 0 --task temporal --loss celoss --opt adamw --LR 0.1 --N_feature 1 --metric temporal_acc --projectname 12_AugmentAnalysis --DEVICE cpu --PATIENCE 100 --augment True --N_feature 1&
python3 mlflow_exp_LearnableFilters.py --N_train 20 --e_train 0.0 --DATASET 13 --SEED 0 --task temporal --loss celoss --opt adamw --LR 0.1 --N_feature 1 --metric temporal_acc --projectname 13_AugmentAnalysis --DEVICE cpu --PATIENCE 100 --augment True --N_feature 1&
python3 mlflow_exp_LearnableFilters.py --N_train 20 --e_train 0.0 --DATASET 14 --SEED 0 --task temporal --loss celoss --opt adamw --LR 0.1 --N_feature 1 --metric temporal_acc --projectname 14_AugmentAnalysis --DEVICE cpu --PATIENCE 100 --augment True --N_feature 1&

wait