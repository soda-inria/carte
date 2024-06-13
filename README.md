# CARTE: <br />Pretraining and Transfer for Tabular Learning

(The repo is still under construction, it will be available as soon a possible)

This repository contains the implementation of the paper CARTE: Pretraining and Transfer for Tabular Learning.

<!-- training a foundation model for tabular data by treating each table row as a star graph and training a graph transformer on top of this representation.
-->

## Installation

Create a new environment with python 3.10, and install the requirements with the requirements.txt file on your environment:

```
git clone https://github.com/soda-inria/carte
cd carte
conda env create -n <env_name> python=3.10
conda activate <env_name>
pip install -r requirements.txt
```

To reproduce the results presented in our paper install additional requirements with optional-requirements.txt file on your environment:

```
pip install -r requirements-optional.txt
```

**Downloading data**

- FastText embeddings: <br/> CARTE uses FastText embeddings which first need to be downloaded in your machine. After setting up the environment,

- Benchmark datasets: <br/>
  The datasets used in our paper can be downloaded

- Large language models: <br/>

- KEN embeddings: <br/>

## Getting started

The best way to get familiar with using CARTE is through the examples.

**Running CARTE for singletables:** <br/>follow through examples/carte_single_tables.ipynb

**Running CARTE for multitables:** <br/>follow through examples/carte_joint_learning.ipynb

## Reproducing results of CARTE paper

We provide the searched parameters for the baselines used for comparisons. Currently, we

## Our paper

```
@article{kim2024carte,
  title={CARTE: pretraining and transfer for tabular learning},
  author={Kim, Myung Jun and Grinsztajn, L{\'e}o and Varoquaux, Ga{\"e}l},
  journal={arXiv preprint arXiv:2402.16785},
  year={2024}
}
```
