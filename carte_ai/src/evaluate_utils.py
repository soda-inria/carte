import numpy as np
import pandas as pd

from ast import literal_eval
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GroupShuffleSplit
from carte_ai.configs.directory import config_directory


def set_split(data, data_config, num_train, random_state):
    """Set train/test split given the random state."""

    target_name = data_config["target_name"]
    X = data.drop(columns=target_name)
    y = data[target_name]
    y = np.array(y)

    if data_config["repeated"]:
        entity_name = data_config["entity_name"]
    else:
        entity_name = np.arange(len(y))

    groups = np.array(data.groupby(entity_name).ngroup())
    num_groups = len(np.unique(groups))
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=int(num_groups - num_train),
        random_state=random_state,
    )
    idx_train, idx_test = next(iter(gss.split(X=y, groups=groups)))

    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    return X_train, X_test, y_train, y_test


def extract_best_params(data_name, method, num_train, random_state):
    """Extract the best parameters in the CARTE paper."""

    if "tabpfn" in method:
        return dict()
    else:
        # Load compiled log
        df_log_dir = f"{config_directory['results']}/compiled_results/results_carte_baseline_bestparams.csv"
        df_log = pd.read_csv(df_log_dir)

        # Obtain the mask
        mask = df_log["data_name"] != data_name
        mask += df_log["model"] != method
        mask += df_log["num_train"] != num_train
        mask += df_log["random_state"] != random_state

        # Extract the best paramameters
        best_params_ = df_log["best_param"].copy()
        best_params = literal_eval(best_params_[~mask].iloc[0])
        return best_params


def set_score_criterion(task):
    """Set scoring method for CV and score criterion in final result."""

    if task == "regression":
        scoring = "r2"
        score_criterion = ["r2", "rmse"]
    else:
        scoring = "roc_auc"
        score_criterion = ["roc_auc", "avg_precision"]
    score_criterion += ["preprocess_time"]
    score_criterion += ["inference_time"]
    score_criterion += ["run_time"]
    return scoring, score_criterion


def shorten_param(param_name):
    """Shorten the param_names for column names in search results."""

    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name


def check_pred_output(y_train, y_pred):
    """Set the output as the mean of train data if it is nan."""

    if np.isnan(y_pred).sum() > 0:
        mean_pred = np.mean(y_train)
        y_pred[np.isnan(y_pred)] = mean_pred
    return y_pred


def reshape_pred_output(y_pred):
    """Reshape the predictive output accordingly."""

    num_pred = len(y_pred)
    if y_pred.shape == (num_pred, 2):
        y_pred = y_pred[:, 1]
    elif y_pred.shape == (num_pred, 1):
        y_pred = y_pred.ravel()
    else:
        pass
    return y_pred


def set_score_criterion(task):
    """Set scoring method for CV and score criterion in final result."""

    if task == "regression":
        scoring = "r2"
        score_criterion = ["r2", "rmse"]
    else:
        scoring = "roc_auc"
        score_criterion = ["roc_auc", "avg_precision"]
    score_criterion += ["preprocess_time"]
    score_criterion += ["inference_time"]
    score_criterion += ["run_time"]
    return scoring, score_criterion


def return_score(y_target, y_pred, task):
    """Return score results for given task."""

    if task == "regression":
        score_r2 = r2_score(y_target, y_pred)
        mse = mean_squared_error(y_target, y_pred)
        score_rmse = np.sqrt(mse)
        return score_r2, score_rmse
    else:
        score_auc = roc_auc_score(y_target, y_pred)
        score_avg_precision = average_precision_score(y_target, y_pred)
        return score_auc, score_avg_precision


def col_names_per_type(data, target_name):
    """Extract column names per type."""
    num_col_names = data.select_dtypes(exclude="object").columns.tolist()
    if target_name in num_col_names:
        num_col_names.remove(target_name)
    cat_col_names = data.select_dtypes(include="object").columns.tolist()
    if target_name in cat_col_names:
        cat_col_names.remove(target_name)
    return num_col_names, cat_col_names
