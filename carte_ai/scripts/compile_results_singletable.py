"""Script for compling results"""

# >>>
if __name__ == "__main__":
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ["PROJECT_DIR"] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

import json
from glob import glob
from carte_ai.configs.directory import config_directory
import numpy as np
import pandas as pd


def _load_config(data_name):
    config_data_dir = (
        f"{config_directory['data_singletable']}/{data_name}/config_data.json"
    )
    filename = open(config_data_dir)
    config_data = json.load(filename)
    filename.close()
    return config_data


if __name__ == "__main__":

    result_dir_base = f"{config_directory['results']}/singletable"
    result_filenames = glob(f"{result_dir_base}/*/*.csv*")

    df_score = pd.DataFrame()
    for path in result_filenames:
        data_name = path.split("/")[-2]
        file_name = path.split("/")[-1]
        method_name = file_name.split(f"{data_name}_")[1].split("_num_train")[0]
        num_train = file_name.split("num_train-")[1].split("_")[0]
        random_state = file_name.split("rs-")[1].split(".csv")[0]

        config_data = _load_config(data_name)
        task = config_data["task"]
        score_measure = "r2" if task == "regression" else "roc_auc"

        score_ = pd.read_csv(path)
        score_col = [col for col in score_.columns if score_measure in col][0]
        score_[score_col].iloc[0]

        df_score_ = dict()
        df_score_["model"] = method_name
        df_score_["score"] = score_[score_col].iloc[0]
        df_score_["data_name"] = data_name
        df_score_["num_train"] = num_train
        df_score_["random_state"] = random_state
        df_score_["task"] = task
        df_score_ = pd.DataFrame([df_score_])

        df_score = pd.concat([df_score, df_score_], axis=0)

    df_score.reset_index(drop=True, inplace=True)
    save_dir = (
        f"{config_directory['compiled_results']}/results_carte_baseline_singletable.csv"
    )
    df_score.to_csv(save_dir, index=False)
