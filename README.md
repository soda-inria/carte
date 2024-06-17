# CARTE: <br />Pretraining and Transfer for Tabular Learning

This repository contains the implementation of the paper CARTE: Pretraining and Transfer for Tabular Learning.

<!-- training a foundation model for tabular data by treating each table row as a star graph and training a graph transformer on top of this representation.
-->

## Installation

**Required dependencies**

CARTE works with PyTorch and python version >=3.10. Create a new environment with python 3.10, and install the appropriate PyTorch version for your machine. Then, install the dependencies with the requirements.txt file on your environment:

```
pip install -r requirements.txt
```

In the requirements.txt file the package `torch_scatter` may depend on the specific PyTorch version. It is recommended to install the appropriate version by changing the first line ('--find-links') to the specific version outlined in https://data.pyg.org/whl/.

To reproduce the results presented in our paper, install additional requirements with requirements-optional.txt file on your environment:

```
pip install -r requirements-optional.txt
```

**Downloading data**

The download of required data (Fasttext, datasets, etc) can be managed by running

```
python scripts/download_data.py -op <option for datasets> -ir <include raw data> -ik <include KEN data>
```

or channge options in bash file and running it with

```
bash scripts/download_data.sh
```

Note that the code will download the FastText embeddings if it is not present under the `data/etc` folder. If the embeddings are stored in a different directory, please change the 'config_directory["fasttext"]' in the `configs/directory.py`

The variables are:

- options (-op): Options to download preprocessed datasets used in our paper.<br/>
  Stored under `data/data_singletable`.

  - "carte" : No downloadings of datasets.
  - "basic_examples" : Download 4 preprocessed datasets for running examples.
  - "full_examples" : Download all 51 preprocessed datasets without the LLM features.
  - "full_benchmark" : Download all 51 preprocessed datasets including the LLM features.

- include_raw (-ir) : Benchmark raw datasets <br/>
  The original datasets without any preprocessing. "True" to download all 51 datasets or "False" otherwise. Stored under `data/data_raw`.

- include_ken (-ik) : KEN (YAGO knowledge graph) embeddings <br/>
  The KEN embeddings, which are knowledge graph embeddings of YAGO entities. "True" to download the embeddings or "False" otherwise. Stored under `data/etc`.

Example (in the prepared environment) downloading FastText embeddings and the 4 datasets for examples for running CARTE:

```
python scripts/download_data.py -op "basic_examples" -ir "False" -ik "False"
```

## Getting started

The best way to get familiar with using CARTE is through the examples. After setting up the datasets, run the following examples if needed.

**Running CARTE for singletables:** <br/>follow through `examples/carte_single_tables.ipynb`

**Running CARTE for multitables:** <br/>follow through `examples/carte_joint_learning.ipynb`

## Reproducing results of CARTE paper

It will be updated as soon as possible.

## Our paper

```
@article{kim2024carte,
  title={CARTE: pretraining and transfer for tabular learning},
  author={Kim, Myung Jun and Grinsztajn, L{\'e}o and Varoquaux, Ga{\"e}l},
  journal={arXiv preprint arXiv:2402.16785},
  year={2024}
}
```
