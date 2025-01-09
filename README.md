# Towards Temporal Information Processing -- Printed Neuromorphic Circuits with Learnable Filters

This github repository is for the paper at NanoArch'23 - Towards Temporal Information Processing -- Printed Neuromorphic Circuits with Learnable Filters

cite as
```
Towards Temporal Information Processing -- Printed Neuromorphic Circuits with Learnable Filters
Zhao, H.; Pal, P.; Hefenbrock, M.; Beigl, M.; Tahoori, M.
Proceedings of the 18th ACM International Symposium on Nanoscale Architectures. 2023.
```



Usage of the code:

1. Training of printed Temporal Processing Neuromorphic Circuit (pTPNC)

~~~
$ sh run_LearnableFilter.sh
~~~

Alternatively, the experiments can be conducted by running command lines in `run_LearnableFilter.sh` separately, e.g.,

~~~
$ sbatch exp_LearnableFilters.py --DATASET 0 --SEED 0 --task temporal --loss celoss --metric temporal_acc --projectname LearnableFilters
$ sbatch exp_LearnableFilters.py --DATASET 0 --SEED 1 --task temporal --loss celoss --metric temporal_acc --projectname LearnableFilters
...
~~~

Additionally, the baselines, e.g., the previsous printed Neuromorphic Circuits (pNCs) can be experimented by running `run_baseline_pNN.sh`.


2.   After training printed neural networks, the trained networks are in `./LearnableFilters/model/`, the log files for training can be found in `./LearnableFilters/log/`. If there is still files in `./LearnableFilters/temp/`, you should run the corresponding command line to train the networks further. Note that, each training is limited to 48 hours, you can change this time limitation in `configuration.py`

Similarly, the baselines can be found in the folder `./Baseline/`.



3.   Evaluation can be done by running the `Evaluation.ipynb` in the corresponding folders for pTPNC or baselines.
