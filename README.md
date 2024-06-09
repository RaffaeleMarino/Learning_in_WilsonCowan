# Repository for Learning in Wilson-Cowan Model for metapopulation

This repository contains all the code and data required to reproduce the results presented in the paper titled "Learning in Wilson-Cowan Model for metapopulation (https://arxiv.org/abs/xxx)" published on arXiv. Below, you will find detailed instructions on how to set up, run, and understand the provided code.

## Table of Contents

- [Introduction](#introduction)
- [Structure of the Repository](#structure-of-the-repository)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
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

- Python 3.10.14
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow 2.13.1

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
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

## Contact

For any questions or issues, please open an issue on this repository or contact the authors:

- [Raffaele Marino](mailto:raffaele.marino@unifi.it)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize the template to better fit your specific needs and the structure of your project.
