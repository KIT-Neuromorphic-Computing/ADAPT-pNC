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
To run the main experiments, use the following command:

```bash
python main.py --config configs/experiment.yaml
```

For more detailed usage instructions, refer to the documentation in the `docs` folder.

## Examples
Here are some example commands to run different parts of the project:

```bash
# Train the model
python train.py --config configs/train.yaml

# Evaluate the model
python evaluate.py --config configs/evaluate.yaml
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