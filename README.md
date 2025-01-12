<h1 align="center">ADAPT-pNC: Mitigating Device Variability and Sensor Noise in Printed Neuromorphic Circuits with SO Adaptive Learnable Filters</h1>

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Examples](#examples)
5. [Contributing](#contributing)
6. [License](#license)
7. [Citation](#citation)

## Introduction
ADAPT-pNC is a project focused on mitigating device variability and sensor noise in printed neuromorphic circuits using Self-Organizing (SO) Adaptive Learnable Filters. This repository contains the code and resources necessary to reproduce the experiments and results presented in our paper.

## Installation
To get started with ADAPT-pNC, clone the repository and install the required dependencies:

```bash
git clone /home/kit/itec/qc0876/projects/ADAPT_pNC
cd ADAPT_pNC
pip install -r requirements.txt
```

## Usage

### Training

In general, you can consider this prompt to train the model:


```bash
python3 exp_LearnableFilters.py --N_train <value> --e_train <value> --DATASET <value> --SEED <value> --task <value> --loss <value> --opt <value> --LR <value> --N_feature <value> --metric <value> --projectname <value> --DEVICE <value> --PATIENCE <value> --NOISE_LEVEL <value> --augment <value> --WARP_FACTOR <value> --SFR_down <value> --SFR_up <value> --CROP_SIZE <value>
```


- If you want to use the normal learnable filter instead of the second order learnable filter, you can change the following line in the `FilterGroup` class in `PrintedLearnableFilter.py`:
    
    Uncomment the line and modify it as needed to switch to the normal learnable filter.

    ```python
    # self.FilterGroup.append(LearnableFilter(args, [betas1[n], betas2[-(n+1)]], random_state))
    ```

- If you want to consider augmentation, set `--aug` to `True` and modify the corresponding hyperparameters:

    ```bash
    python main.py ... --aug true --max_drift <value> --n_speed_change <value> --scale <value> --max_speed_ratio <value>
    ```
    
- To change the variation setup during training, you can adjust the following parameters:

    - `--e_train` (float): The variation rate for training. Modify this parameter to change the variation rate.
    - `--N_train` (int): The number of samples for Monte Carlo sampling during training. Adjust this parameter to change the number of samples used in Monte Carlo sampling.

For more information, you can check out the `configuration.py` file.

### Inference

For evaluating the trained model, you can go to the `Evaluation.ipynb` notebook and change the parameters or model addresses based on your preference.

### Hyperparameter tuning via MLFlow

We have obtained the best possible values for data augmentation hyperparameters. The code exists in the `mlflow_exp_LearnableFilter.py` file.



## Examples
Here are some example commands to run different parts of the project:

```bash
# Train the model
python3 exp_LearnableFilters.py --N_train 20 --e_train 0.1 --DATASET 0 --SEED 0 --task temporal --loss celoss --opt adamw --LR 0.1 --N_feature 1 --metric temporal_acc --projectname 10_VariationAnalysisNormal --DEVICE cpu --PATIENCE 100 --NOISE_LEVEL 0.06497824538 --augment True --WARP_FACTOR 0.1666569224 --SFR_down 0.8474484242 --SFR_up 1.076289025 --CROP_SIZE 50.0
```

## Contributing
We welcome contributions to the ADAPT-pNC project. Please read our [contributing guidelines](CONTRIBUTING.md) for more information on how to get involved.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation
If you use this code or data in your research, please cite our paper:

```
@article{your_paper,
    title={ADAPT-pNC: Mitigating Device Variability and Sensor Noise in Printed Neuromorphic Circuits with SO Adaptive Learnable Filters},
    author={Your Name and Co-authors},
    journal={Journal Name},
    year={2023},
    volume={X},
    number={Y},
    pages={Z-ZZ},
    doi={DOI}
}
```