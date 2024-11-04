""" Python script for preparing datasets for evaluation
"""

# >>>
if __name__ == "__main__":
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ["PROJECT_DIR"] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

import pandas as pd
import numpy as np
import pickle
import json
import os
from carte_ai.configs.directory import config_directory
from carte_ai.configs.carte_configs import carte_datalist
from carte_ai.src.preprocess_utils import (
    extract_fasttext_features,
    extract_llm_features,
    table2llmfeatures,
)


def data_preprocess(data_name: str, device: str = "cuda:0"):

    # Load data
    data_pd_dir = f"{config_directory['data_singletable']}/{data_name}/raw.parquet"
    data_pd = pd.read_parquet(data_pd_dir)
    data_pd.fillna(value=np.nan, inplace=True)

    # Basic settings for the data
    config_data_dir = (
        f"{config_directory['data_singletable']}/{data_name}/config_data.json"
    )
    filename = open(config_data_dir)
    config_data = json.load(filename)

    # Set the data without the target
    data_X = data_pd.drop(columns=config_data["target_name"])

    data = dict()
    data_fasttext = None
    data_llm = None
    data_sentence_llm_embed_num = None
    data_sentence_llm_concat_num = None

    if config_data["entity_name"] is not None:
        data_fasttext = extract_fasttext_features(
            data=data_X,
            extract_col_name=config_data["entity_name"],
        )
        data_llm = extract_llm_features(
            data=data_X,
            extract_col_name=config_data["entity_name"],
            device=device,
        )
    else:
        pass

    data_sentence_llm_embed_num = table2llmfeatures(
        data=data_X,
        embed_numeric=True,
        device=device,
    )
    data_sentence_llm_concat_num = table2llmfeatures(
        data=data_X,
        embed_numeric=False,
        device=device,
    )

    data["fasttext"] = data_fasttext
    data["llm"] = data_llm
    data["sentence-llm-embed-num"] = data_sentence_llm_embed_num
    data["sentence-llm-concat-num"] = data_sentence_llm_concat_num

    save_dir = f"{config_directory['data_singletable']}/{data_name}/external.pickle"

    with open(save_dir, "wb") as pickle_file:
        pickle.dump(data, pickle_file)


def main(datalist, device: str = "cuda:0"):

    datalist_total = carte_datalist

    # Setting methods
    if "all" in datalist:
        data_list = datalist_total
    else:
        data_list = datalist
        if isinstance(data_list, list) == False:
            data_list = list(data_list)

    for data_name in data_list:
        data_preprocess(data_name=data_name, device=device)
        print(f"{data_name} complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Preparation")
    parser.add_argument(
        "-dt",
        "--datalist",
        nargs="+",
        type=str,
        help="List of data",
    )
    parser.add_argument(
        "-de",
        "--device",
        type=str,
        help="Device",
    )
    args = parser.parse_args()

    main(args.datalist, args.device)
