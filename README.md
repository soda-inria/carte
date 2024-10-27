[![Downloads](https://img.shields.io/pypi/dm/carte-ai)](https://pypi.org/project/carte-ai/)
[![PyPI Version](https://img.shields.io/pypi/v/carte-ai)](https://pypi.org/project/carte-ai/)
[![Python Version](https://img.shields.io/pypi/pyversions/carte-ai)](https://pypi.org/project/carte-ai/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)


# CARTE: <br />Pretraining and Transfer for Tabular Learning

![CARTE_outline](carte_ai/data/etc/outline_carte.jpg)

This repository contains the implementation of the paper CARTE: Pretraining and Transfer for Tabular Learning.

CARTE is a pretrained model for tabular data by treating each table row as a star graph and training a graph transformer on top of this representation.

## Colab Examples (Give it a test):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PeltEmNLehQ26VQtFJhl7OxnzCS8rPMT?usp=sharing)
* CARTERegressor on Wine Poland dataset
* CARTEClassifier on Spotify dataset
  


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

or change the options in bash file and running it with

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
  The original datasets without any preprocessing. "True" to download all 51 datasets or "False" otherwise. Stored under `data/data_raw`. See `scripts/preprocess_raw.py` for specific details on preprocessing.

- include_ken (-ik) : KEN (YAGO knowledge graph) embeddings <br/>
  The KEN embeddings, which are knowledge graph embeddings of YAGO entities. "True" to download the embeddings or "False" otherwise. Stored under `data/etc`.

Example (in the prepared environment) downloading FastText embeddings and the 4 datasets for examples for running CARTE:

```
python scripts/download_data.py -op "basic_examples" -ir "False" -ik "False"
```

The datasets can also be found https://huggingface.co/datasets/inria-soda/carte-benchmark.

## Getting started

The best way to get familiar with using CARTE is through the examples. After setting up the datasets, run the following examples if needed.

**Running CARTE for singletables:** <br/>follow through `examples/1. carte_single_tables.ipynb`

**Running CARTE for multitables:** <br/>follow through `examples/2. carte_joint_learning.ipynb`

<em>Note: To run through the examples, it is recommended to have at least 64GB of RAM for single tables and 128GB for multitables. We are currently working to reduce the memory consumption.</em>

## Reproducing results of CARTE paper

Currently, we provide codes for generating results for singletables. The updates for reproducing results on multitables will be updated.

To generate results for singletables, run: 

```
python scripts/evaluate_singletable.py -dn <data name> -nt <train size> -m <method to evaluation> -rs <random state values> -b <include bagging> -dv <device to run>
```

The variables are:

- data_name (-dn): Name of the dataset.<br/>
  specific name under the `data/data_singletable` folder or "all" to run all the datasets. 

- num_train (-nt) : Train size to evaluate. <br/>
  "all" to run train sizes of {32, 64, 128, 256, 512, 1024, 2048}.

- method (-m) : Method to evaluate (See carte_singletable_baselines in `configs/carte_configs`)<br/>
  - "full" : the full list of all baselines (See `carte_singletable_baselines['full']`).
  - "reduced" : the reduced list of all baselines in CARTE paper (See `carte_singletable_baselines['reduced']`).
  - "f-r" : the list of baselines excluding the reduced list from the full list.
  - "any other method" : any other method in `carte_singletable_baselines['full']`.

- random_state (-rs) : Random state value. <br/>
  "all" to run train sizes of {1, 2, 3, ..., 10}

- bagging (-b) : Indicate to include the bagging strategy or not. <br/>
  "True" to include the bagging strategy in analysis. Note that for neural-networks based models, it runs the bagging strategy even when it is set to "False".

- device (-dv) :  <br/>
  "cpu" to run on cpus or "cuda" to run on gpus. Requires some specifications if ran on gpus.

Example running the 'wina poland' dataset with train size of 128 and random state 1 in the examples:
```
python scripts/evaluate_singletable.py -dn "wina_pl" -nt "128" -m "reduced" -rs "1" -b "False" -dv "cpu"
```
Running this will create a folder `results/singletable/wina_pl`, in which the results of each baseline will be stored as a csv file.

After obtaining the results under the `results/singletable` folder, run `scripts/compile_results_singletable.py` to compile results as a single dataframe, which will be saved as a csv file, named 'results_carte_baseline_singletable.csv', in the `results/compiled_results` folder. 

Then, follow through `examples/3. carte_singletable_visualization` for visualization of the results.

<em>The script does not run the random search (as done in the CARTE paper). To ease the computations and visualization, we provide the parameters for each baselines found from the random search. However, running the total comparison may take a long time, and it is recommended to run them on a parallel computing machines (e.g., clusters). The evaluation script only shows the guidelines for reproducing the results and modifications for parallelization suitable for each use-case should be made. For visualization purposes, we also provide the compiled results. </em>

## Our paper

```
@article{kim2024carte,
  title={CARTE: pretraining and transfer for tabular learning},
  author={Kim, Myung Jun and Grinsztajn, L{\'e}o and Varoquaux, Ga{\"e}l},
  journal={arXiv preprint arXiv:2402.16785},
  year={2024}
}
```
