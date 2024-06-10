# Repository for Learning in Wilson-Cowan Model for metapopulation

This repository contains all the code and data required to reproduce the results presented in the paper titled "Learning in Wilson-Cowan Model for metapopulation (https://arxiv.org/abs/xxx)" published on arXiv. Below, you will find detailed instructions on how to set up, run, and understand the provided code.

![cartoondynamics](https://github.com/RaffaeleMarino/Learning_in_WilsonCowan/assets/44016352/7b9ae2ae-fff3-48a4-a72b-368588b14169)

## Table of Contents

- [Introduction](#introduction)
- [Structure of the Repository](#structure-of-the-repository)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Experiment Results](#experiment-results)
- [Contact](#contact)
- [License](#license)


## Introduction

This repository contains the implementation of the algorithms and experiments described in the paper "Learning in Wilson-Cowan Model for metapopulation (https://arxiv.org/abs/xxx)". The goal is to provide  a transparent and reproducible set of tools to verify and extend the findings of our work.

## Structure of the Repository

The repository is structured as follows:

- `src/`: Source code for the main algorithms and models.
- `src/WC`: Source code for Wilsong-Cowan model for metapopulation.
- `src/CNNandWC`: Source code for Wilsong-Cowan model for metapopulation in combination with CNN.
- `src/BERTandWC`: Source code for Wilsong-Cowan model for metapopulation in combination with BERT.
  
In each directory, there is a folder containing the specific dataset being analyzed.
## Requirements

To run the code, you will need the following software and libraries:

- Python 3.12.3 and 3.10.14 (for BERT) 
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow 2.16.1 and 2.13.1 (for BERT)

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```
For experiments with BERT :

```bash
pip install -r requirementsBERT.txt
```

## Installation

Clone the repository to your local machine using the following command:

```bash
https://github.com/RaffaeleMarino/Learning_in_WilsonCowan.git
```

## Usage

To run the main experiments, use the following command:

```bash
cd src/
cd *
cd **
python main_**.py
```
where * identifies the name of the architecture, while ** identifies the name of the dataset.

# Experiment Results

| Dataset        | Model        | Hyperparameters                                         | Results          |
|----------------|------------------------------------------------------------------------|------------------|
| MNIST          |              | Learning Rate: 0.01<br>Batch Size: 32<br>Epochs: 10     | Accuracy: 98%    |
| FASHION MNIST  |              | Learning Rate: 0.001<br>Batch Size: 64<br>Epochs: 20    | Accuracy: 91%    |
| CIFAR10        |              | Learning Rate: 0.001<br>Batch Size: 128<br>Epochs: 50   | Accuracy: 85%    |
| TF-FLOWERS     |              | Learning Rate: 0.0001<br>Batch Size: 32<br>Epochs: 30   | Accuracy: 78%    |
| IMDB           |              | Learning Rate: 0.01<br>Batch Size: 16<br>Epochs: 15     | Accuracy: 87%    |

| Dataset        | Learning Rate | Batch Size | Epochs | Accuracy |
|----------------|---------------|------------|--------|----------|
| MNIST          | 0.01          | 32         | 10     | 98%      |
| FASHION MNIST  | 0.001         | 64         | 20     | 91%      |
| CIFAR10        | 0.001         | 128        | 50     | 85%      |
| TF-FLOWERS     | 0.0001        | 32         | 30     | 78%      |
| IMDB           | 0.01          | 16         | 15     | 87%      |

## Notes on Hyperparameters
- **Learning Rate**: The learning rate used for the optimizer.
- **Batch Size**: The batch size used during training.
- **Epochs**: The number of epochs the model was trained for.

## Notes on Hyperparameters
- **Learning Rate**: The learning rate used for the optimizer.
- **Batch Size**: The batch size used during training.
- **Epochs**: The number of epochs the model was trained for.


## Contact

For any questions or issues, please open an issue on this repository or contact the authors:

- [Raffaele Marino](mailto:raffaele.marino@unifi.it)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
