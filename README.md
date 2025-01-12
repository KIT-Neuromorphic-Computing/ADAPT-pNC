<h1 align="center">ADAPT-pNC: Mitigating Device Variability and Sensor Noise in Printed Neuromorphic Circuits with SO Adaptive Learnable Filters</h1>

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Examples](#examples)
6. [Results](#results)
7. [Citation](#citation)

## Introduction
ADAPT-pNC is a robust framework for printed neuromorphic circuits (pNCs) that addresses challenges of device variability and sensor noise. By integrating variation-aware second-order learnable filters (SO-LFs), data augmentation, and advanced training techniques, ADAPT-pNC achieves high accuracy and robustness in temporal data processing tasks.

This repository provides the code, models, and resources to reproduce the results from our **DATE 2025** paper, where ADAPT-pNC demonstrated significant improvements over traditional methods in mitigating variability in printed electronics.

## Features
- **Second-Order Learnable Filters (SO-LF):** Enhanced dynamic response for robust temporal data processing.
- **Variation-Aware Training:** Models component and data variations to ensure reliability.
- **Data Augmentation:** Techniques like jittering, time warping, and magnitude scaling simulate real-world sensor variations.
- **Benchmarked Performance:** Validated on 15 time-series datasets with state-of-the-art results.

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/KIT-Neuromorphic-Computing/ADAPT-pNC.git
cd ADAPT-pNC
pip install -r requirements.txt
```

## Usage

### Training

Train the model using:


```bash
python3 exp_LearnableFilters.py \
    --N_train <value> \
    --e_train <value> \
    --DATASET <value> \
    --SEED <value> \
    --task <value> \
    --loss <value> \
    --opt <value> \
    --LR <value> \
    --N_feature <value> \
    --metric <value> \
    --projectname <value> \
    --DEVICE <value> \
    --PATIENCE <value> \
    --NOISE_LEVEL <value> \
    --augment <value> \
    --WARP_FACTOR <value> \
    --SFR_down <value> \
    --SFR_up <value> \
    --CROP_SIZE <value>

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

For evaluating the trained model, you can go to the `Evaluation.ipynb` notebook and change the parameters or model paths as needed.

### Hyperparameter Tuning

We have obtained the best possible values for data augmentation hyperparameters. The code exists in the `mlflow_exp_LearnableFilter.py` file.



## Examples
Train the model with default settings for dataset `0`:

```bash
# Train the model
python3 exp_LearnableFilters.py --N_train 20 --e_train 0.1 --DATASET 0 --SEED 0 --task temporal --loss celoss --opt adamw --LR 0.1 --N_feature 1 --metric temporal_acc --projectname 10_VariationAnalysisNormal --DEVICE cpu --PATIENCE 100 --NOISE_LEVEL 0.06497824538 --augment True --WARP_FACTOR 0.1666569224 --SFR_down 0.8474484242 --SFR_up 1.076289025 --CROP_SIZE 50.0
```

## Results

ADAPT-pNC has been extensively tested on 15 benchmark time-series datasets. Here’s a summary of the key results:

- **Accuracy Improvement:** ~45% over baseline pTPNCs.
- **Power Efficiency:** ~90% reduction in power consumption.
- **Hardware Trade-off:** ~1.9× increase in device count for significant robustness gains.

| Dataset      | Elman RNN | Baseline pTPNC | ADAPT-pNC |
|--------------|-----------|----------------|-----------|
| CBF          | 0.683     | 0.615          | **0.877** |
| PowerCons    | 0.651     | 0.797          | **0.004** |
| SmoothS      | 0.447     | 0.653          | **0.864** |
| Symbols      | 0.141     | 0.369          | **0.697** |
| ...          | ...       | ...            | ...       |
| **Average**  | 0.501     | 0.582          | **0.726** |

For detailed results, refer to **Table I** in our paper.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{gheshlaghi2025adaptpnc,
    title={ADAPT-pNC: Mitigating Device Variability and Sensor Noise in Printed Neuromorphic Circuits with SO Adaptive Learnable Filters},
    author={Tara Gheshlaghi and Priyanjana Pal and Haibin Zhao and Michael Hefenbrock and Michael Beigl and Mehdi B. Tahoori},
    booktitle={Design, Automation and Test in Europe Conference (DATE)},
    year={2025},
    organization={IEEE}
}
