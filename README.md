# Repository for Learning in Wilson-Cowan Model for metapopulation

This repository contains all the code and data required to reproduce the results presented in the paper titled "Learning in Wilson-Cowan Model for metapopulation" published on arXiv (https://arxiv.org/xxx). Below, you will find detailed instructions on how to set up, run, and understand the provided code.

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

This repository contains the implementation of the algorithms and experiments described in the paper "Learning in Wilson-Cowan Model for metapopulation". The goal is to provide  a transparent and reproducible set of tools to verify and extend the findings of our work.

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


| Dataset        | Model           | Learning Rate | Batch Size | Epochs  | Accuracy | SOTA     |
|----------------|-----------------|---------------|------------|---------|----------|----------|
| MNIST          | WC              | 0.1           | 200        | 525     | 98.18%   | 99.87%[1]|
| FASHION MNIST  | WC              | 0.1           | 200        | 350     | 88.39%   | 96.91%[2]|
| MNIST          | CNN & WC        | 0.0001        | 10 & 200   | 35 & 70 | 99.31%   | 99.87%[1]|
| FASHION MNIST  | CNN & WC        | 0.0001        | 10 & 200   | 35 & 70 | 91.35%   | 96.91%[2]|
| CIFAR10        | CNN & WC        | 0.0001        | 10 & 200   | 70 & 70 | 85.39%   | 99.50%[3]|
| TF-FLOWERS     | CNN & WC        | 0.001         | 10 & 32    | 70 & 100| 83.54%   | 98.00%[4]|
| IMDB           | BERT & WC       | 0.00003       | 4          | 10      | 87.46%   | 96.68%[5]|

## Notes on Hyperparameters
- **Model**: The model architecture used for the experiment.
- **Learning Rate**: The learning rate used for the optimizer.
- **Batch Size**: The batch size used during training.
- **Epochs**: The number of epochs the model was trained for.
- **SOTA**: State-Of-The-Art
- [1] A. Byerly, T. Kalganova, and I. Dear, No routing needed between capsules, Neurocomputing 463, 545 (2021)
- [2] M. S. Tanveer, M. U. K. Khan, and C.-M. Kyung, Fine-tuning darts for image classification, in 2020 25th International Conference on Pattern Recognition (ICPR)(IEEE, 2021) 
- [3] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, An image is worth 16x16 words: Transformers for image recognition at scale, in International Conference on Learning Representations (2021)
- [4] A. Kolesnikov, L. Beyer, X. Zhai, J. Puigcerver, J. Yung, S. Gelly, and N. Houlsby, Big transfer (bit): General visual representation learning in Computer Vision –
ECCV 2020 (2020) 
- [5] B. Csanady, L. Muzsai, P. Vedres, Z. Nadasdy, and A. Lukacs, Llambert: Large-scale low-cost data annotation in NLP, arXiv preprint arXiv:2403.15938 (2024)


## Contact

For any questions or issues, please open an issue on this repository or contact the authors:

- [Raffaele Marino](mailto:raffaele.marino@unifi.it)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
