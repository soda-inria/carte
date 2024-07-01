"""Script for evalutating a model of choice in parallel for singletables.
It uses submitit, a python library to launch programatically SLURM computation.
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

import os
import pickle
import json
import pandas as pd
import numpy as np
import copy
import time
import submitit

from joblib import Memory
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from configs.directory import config_directory
from configs.carte_configs import carte_datalist, carte_singletable_baselines
from src.evaluate_utils import *
from src.carte_estimator import CARTERegressor, CARTEClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
from tabpfn import TabPFNClassifier
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
    BaggingRegressor,
    BaggingClassifier,
)
from sklearn.linear_model import Ridge, LogisticRegression
from src.baseline_singletable_nn import (
    MLPRegressor,
    MLPClassifier,
    RESNETRegressor,
    RESNETClassifier,
)

mem = Memory(location="__cache__", verbose=0)


def _load_data(data_name):
    """Load data, external data, and configs."""

    data_dir = f"{config_directory['data_singletable']}/{data_name}/raw.parquet"
    data_additional_dir = (
        f"{config_directory['data_singletable']}/{data_name}/external.pickle"
    )
    data = pd.read_parquet(data_dir)
    data.fillna(value=np.nan, inplace=True)
    with open(data_additional_dir, "rb") as pickle_file:
        data_additional = pickle.load(pickle_file)
    config_data_dir = (
        f"{config_directory['data_singletable']}/{data_name}/config_data.json"
    )
    filename = open(config_data_dir)
    config_data = json.load(filename)
    filename.close()
    return data, data_additional, config_data


def _prepare_carte_gnn(
    data,
    data_config,
    num_train,
    random_state,
):
    """Preprocess for CARTE (graph construction)."""

    from src.carte_table_to_graph import Table2GraphTransformer

    data_ = data.copy()
    X_train, X_test, y_train, y_test = set_split(
        data_,
        data_config,
        num_train,
        random_state=random_state,
    )
    preprocessor = Table2GraphTransformer()
    X_train = preprocessor.fit_transform(X_train, y=y_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test, y_train, y_test


def _prepare_catboost(
    data,
    data_config,
    num_train,
    random_state,
):
    """Preprocess for CatBoost."""

    data_ = data.copy()
    _, cat_col_names = col_names_per_type(data, data_config["target_name"])
    data_cat = data_[cat_col_names]
    data_cat = data_cat.replace(np.nan, "nan", regex=True)
    data_[cat_col_names] = data_cat
    for col in cat_col_names:
        data_[col] = data_[col].astype("category")
    X_train, X_test, y_train, y_test = set_split(
        data_,
        data_config,
        num_train,
        random_state=random_state,
    )
    # index of categorical columns
    cat_features = [X_train.columns.get_loc(col) for col in cat_col_names]
    return (
        np.array(X_train),
        np.array(X_test),
        np.array(y_train),
        np.array(y_test),
        cat_features,
    )


def _prepare_tablevectorizer(
    data,
    data_config,
    num_train,
    random_state,
    estim_method,
):
    """Preprocess with Tablevectorizer."""

    from skrub import TableVectorizer

    data_ = data.copy()
    X_train, X_test, y_train, y_test = set_split(
        data_,
        data_config,
        num_train,
        random_state=random_state,
    )
    num_col_names, cat_col_names = col_names_per_type(data, data_config["target_name"])

    # Set preprocessors for categorical and numerical
    categorical_preprocessor = TableVectorizer(auto_cast=False, sparse_threshold=0)
    numerical_preprocessor = SimpleImputer(strategy="mean")

    # Set final pipeline for preprocessing depending on the method
    tree_based_methods = ["xgb", "histgb", "randomforest"]
    if estim_method in tree_based_methods:
        preprocessor_final = ColumnTransformer(
            [
                ("numerical", "passthrough", num_col_names),
                ("categorical", categorical_preprocessor, cat_col_names),
            ]
        )
    elif estim_method in ["tabpfn"]:
        preprocessor = ColumnTransformer(
            [
                ("numerical", numerical_preprocessor, num_col_names),
                ("categorical", categorical_preprocessor, cat_col_names),
            ]
        )
        preprocessor_final = Pipeline(
            [
                ("preprocess", preprocessor),
                ("missing", SimpleImputer(strategy="mean")),
            ]
        )
    else:
        preprocessor = ColumnTransformer(
            [
                ("numerical", numerical_preprocessor, num_col_names),
                ("categorical", categorical_preprocessor, cat_col_names),
            ]
        )
        preprocessor_final = Pipeline(
            [
                ("preprocess", preprocessor),
                ("minmax", MinMaxScaler()),
                ("missing", SimpleImputer(strategy="mean")),
            ]
        )
    X_train = preprocessor_final.fit_transform(X_train, y=y_train)
    X_test = preprocessor_final.transform(X_test)

    if estim_method in ["tabpfn"]:
        if X_train.shape[1] > 100:
            n_components = np.min([X_train.shape[0], 100])
            pca_ = PCA(n_components=n_components, svd_solver="full")
            X_train = pca_.fit_transform(X_train)
            X_test = pca_.transform(X_test)

    return X_train, X_test, y_train, y_test


def _prepare_target_encoder(
    data,
    data_config,
    num_train,
    random_state,
    estim_method,
):
    """Preprocess with Target Encoder."""

    data_ = data.copy()
    X_train, X_test, y_train, y_test = set_split(
        data_,
        data_config,
        num_train,
        random_state=random_state,
    )
    num_col_names, cat_col_names = col_names_per_type(data, data_config["target_name"])
    if data_config["task"] == "regression":
        target_type = "continuous"
    else:
        target_type = "binary"

    # Set preprocessors for categorical and numerical
    categorical_preprocessor = TargetEncoder(
        categories="auto",
        target_type=target_type,
        random_state=random_state,
    )
    numerical_preprocessor = SimpleImputer(strategy="mean")

    # Set final pipeline for preprocessing depending on the method
    tree_based_methods = ["xgb", "histgb", "randomforest"]
    if estim_method in tree_based_methods:
        preprocessor_final = ColumnTransformer(
            [
                ("numerical", "passthrough", num_col_names),
                ("categorical", categorical_preprocessor, cat_col_names),
            ]
        )
    elif estim_method in ["tabpfn"]:
        preprocessor_final = ColumnTransformer(
            [
                ("numerical", numerical_preprocessor, num_col_names),
                ("categorical", categorical_preprocessor, cat_col_names),
            ]
        )
    else:
        preprocessor = ColumnTransformer(
            [
                ("numerical", numerical_preprocessor, num_col_names),
                ("categorical", categorical_preprocessor, cat_col_names),
            ]
        )
        preprocessor_final = Pipeline(
            [
                ("preprocess", preprocessor),
                ("minmax", MinMaxScaler()),
            ]
        )
    X_train = preprocessor_final.fit_transform(X_train, y=y_train)
    X_test = preprocessor_final.transform(X_test)

    if estim_method in ["tabpfn"]:
        if X_train.shape[1] > 100:
            n_components = np.min([X_train.shape[0], 100])
            pca_ = PCA(n_components=n_components, svd_solver="full")
            X_train = pca_.fit_transform(X_train)
            X_test = pca_.transform(X_test)

    return X_train, X_test, y_train, y_test


def _prepare_llm(
    data,
    data_config,
    num_train,
    random_state,
):
    """Prepare the llm data. It loads the preprocessed data."""

    data_ = data.copy()
    data_.drop(columns=data_config["entity_name"], inplace=True)
    X_train, X_test, y_train, y_test = set_split(
        data_,
        data_config,
        num_train,
        random_state,
    )

    col_llm, col_not_llm = X_train.columns[:1024], X_train.columns[1024:]

    X_train_llm, X_train_ = X_train[col_llm], X_train[col_not_llm]
    X_test_llm, X_test_ = X_test[col_llm], X_test[col_not_llm]

    if num_train > 1024:
        pca = PCA().set_output(transform="pandas")
        reduced_data_train = pca.fit_transform(X_train_llm)
        dim_reduce_ = np.where(np.cumsum(pca.explained_variance_ratio_) > 0.9)[0][0]
        dim_reduce = min(dim_reduce_, 300)
        reduced_data_train = reduced_data_train.iloc[:, : dim_reduce + 1]
        reduced_data_test = pca.transform(X_test_llm).iloc[:, : dim_reduce + 1]
        X_train = pd.concat([reduced_data_train, X_train_], axis=1)
        X_train = X_train.to_numpy().astype(np.float32)
        X_test = pd.concat([reduced_data_test, X_test_], axis=1)
        X_test = X_test.to_numpy().astype(np.float32)

    return X_train, X_test, y_train, y_test


def _assign_estimator(
    estim_method,
    task,
    device,
    cat_features,
    bagging,
):
    """Assign the specific estimator to train model."""

    # Set number of models for NN-based methods
    if bagging:
        num_model = 1
    else:
        num_model = 15

    if estim_method == "carte-gnn":
        fixed_params = dict()
        fixed_params["batch_size"] = 16
        fixed_params["num_model"] = num_model
        fixed_params["device"] = device
        fixed_params["n_jobs"] = num_model
        fixed_params["random_state"] = 0
        if task == "regression":
            estimator = CARTERegressor(**fixed_params)
        else:
            estimator = CARTEClassifier(**fixed_params)
    elif estim_method == "catboost":
        fixed_params = dict()
        fixed_params["cat_features"] = cat_features
        fixed_params["verbose"] = False
        fixed_params["allow_writing_files"] = False
        fixed_params["thread_count"] = 1
        fixed_params["leaf_estimation_iterations"] = 1
        fixed_params["max_ctr_complexity"] = 1
        if task == "regression":
            estimator = CatBoostRegressor(**fixed_params)
        else:
            estimator = CatBoostClassifier(**fixed_params)
    elif estim_method == "xgb":
        fixed_params = dict()
        fixed_params["booster"] = "gbtree"
        fixed_params["tree_method"] = "exact"  # exact approx hist
        if task == "regression":
            estimator = XGBRegressor(**fixed_params)
        else:
            estimator = XGBClassifier(**fixed_params)
    elif estim_method == "histgb":
        fixed_params = dict()
        if task == "regression":
            estimator = HistGradientBoostingRegressor(**fixed_params)
        else:
            estimator = HistGradientBoostingClassifier(**fixed_params)
    elif estim_method == "randomforest":
        fixed_params = dict()
        if task == "regression":
            estimator = RandomForestRegressor(**fixed_params)
        else:
            estimator = RandomForestClassifier(**fixed_params)
    elif estim_method == "ridge":
        fixed_params = dict()
        estimator = Ridge(**fixed_params)
    elif estim_method == "logistic":
        fixed_params = dict()
        estimator = LogisticRegression(**fixed_params)
    elif estim_method == "mlp":
        fixed_params = dict()
        fixed_params["num_model"] = num_model
        fixed_params["device"] = device
        fixed_params["n_jobs"] = num_model
        fixed_params["random_state"] = 0
        if task == "regression":
            estimator = MLPRegressor(**fixed_params)
        else:
            estimator = MLPClassifier(**fixed_params)
    elif estim_method == "resnet":
        fixed_params = dict()
        fixed_params["num_model"] = num_model
        fixed_params["device"] = device
        fixed_params["n_jobs"] = num_model
        fixed_params["random_state"] = 0
        if task == "regression":
            estimator = RESNETRegressor(**fixed_params)
        else:
            estimator = RESNETClassifier(**fixed_params)
    elif estim_method == "tabpfn":
        estimator = TabPFNClassifier()
    return estimator


def _assign_bagging_estimator(estimator_base, estim_method, task):
    """Assign the bagging estimator if bagging set to true."""

    bagging_estimator = copy.deepcopy(estimator_base)
    if estim_method in ["carte-gnn", "mlp", "resnet"]:
        fixed_params = dict()
        fixed_params["num_model"] = 15
        fixed_params["n_jobs"] = 15
        bagging_estimator.__dict__.update(fixed_params)
    else:
        bagging_params = dict()
        bagging_params["estimator"] = estimator_base
        bagging_params["n_estimators"] = 15
        bagging_params["max_samples"] = 0.8
        bagging_params["n_jobs"] = 15
        bagging_params["random_state"] = 0
        if task == "regression":
            bagging_estimator = BaggingRegressor(**bagging_params)
        else:
            bagging_estimator = BaggingClassifier(**bagging_params)

    return bagging_estimator


# Run evaluation
def run_model(
    data_name,
    num_train,
    method,
    random_state,
    bagging,
    device,
):
    """Run model for specific experiment setting."""

    # Load data
    data, data_additional, data_config = _load_data(data_name)

    # Basic settings
    target_name = data_config["target_name"]
    entity_name = data_config["entity_name"]
    task = data_config["task"]
    _, result_criterion = set_score_criterion(task)
    cat_features = None  # overriden by prepare_... functions if needed

    # Set methods
    method_parse = method.split("_")
    estim_method = method_parse[-1]
    preprocess_method = method_parse[0]

    # Stop for exceptions - Regression/Classification only methods, tabpfn > 1024
    reg_only_methods = ["tablevectorizer_ridge", "target-encoder_ridge"]
    cls_only_methods = [
        method for method in carte_singletable_baselines["full"] if "tabpfn" in method
    ]
    cls_only_methods += [
        method for method in carte_singletable_baselines["full"] if "logistic" in method
    ]
    if (data_config["task"] == "regression") and (method in cls_only_methods):
        return None
    elif (data_config["task"] == "classification") and (method in reg_only_methods):
        return None
    elif (num_train > 1024) and (estim_method == "tabpfn"):
        return None

    # Prepare data
    if "fasttext" in preprocess_method:
        data_fasttext = data_additional["fasttext"].copy()
        data_fasttext.drop_duplicates(subset=entity_name, inplace=True)
        data = data.merge(right=data_fasttext, how="left", on=entity_name)
    elif "llm" in preprocess_method:
        if preprocess_method.split("-")[0] == "sentence":
            data_ = data_additional[preprocess_method].copy()
            data = pd.concat([data_, data[[target_name, entity_name]]], axis=1)
            data.dropna(subset=target_name, inplace=True)
        else:
            data_llm = data_additional["llm"].copy()
            data_llm.drop_duplicates(subset=entity_name, inplace=True)
            data = data.merge(right=data_llm, how="left", on=entity_name)
    else:
        pass

    # Preprocess data
    if "carte-gnn" in preprocess_method:
        X_train, X_test, y_train, y_test = _prepare_carte_gnn(
            data,
            data_config,
            num_train,
            random_state,
        )
    elif "catboost" in preprocess_method:
        X_train, X_test, y_train, y_test, cat_features = _prepare_catboost(
            data,
            data_config,
            num_train,
            random_state,
        )
    elif "tablevectorizer" in preprocess_method:
        X_train, X_test, y_train, y_test = _prepare_tablevectorizer(
            data,
            data_config,
            num_train,
            random_state,
            estim_method,
        )
    elif "target-encoder" in preprocess_method:
        X_train, X_test, y_train, y_test = _prepare_target_encoder(
            data,
            data_config,
            num_train,
            random_state,
            estim_method,
        )
    elif "llm" in preprocess_method:
        X_train, X_test, y_train, y_test = _prepare_llm(
            data,
            data_config,
            num_train,
            random_state,
        )

    # Assign estimators
    best_params = extract_best_params(data_name, method, num_train, random_state)
    estimator = _assign_estimator(
        estim_method,
        task,
        device,
        cat_features,
        bagging,
    )
    estimator.__dict__.update(best_params)
    estimator_bagging = _assign_bagging_estimator(estimator, estim_method, task)

    # Create directory for saving results
    result_save_dir_base = f"{config_directory['results']}/singletable/{data_name}"
    if not os.path.exists(result_save_dir_base):
        os.makedirs(result_save_dir_base, exist_ok=True)

    # Run without bagging strategy
    marker = f"{data_name}_{method}_num_train-{num_train}_rs-{random_state}"
    results_model_dir = result_save_dir_base + f"/{marker}.csv"

    # Do not run if result already exists
    if os.path.exists(results_model_dir):
        pass
    else:
        estimator.fit(X_train, y_train)
        if task == "regression":
            y_pred = estimator.predict(X_test)
        else:
            y_pred = estimator.predict_proba(X_test)
            y_pred = reshape_pred_output(y_pred)
        y_pred = check_pred_output(y_train, y_pred)
        score = return_score(y_test, y_pred, task)

        results_ = dict()
        results_[result_criterion[0]] = score[0]
        results_[result_criterion[1]] = score[1]
        results_model = pd.DataFrame([results_], columns=result_criterion[:2])
        results_model.columns = f"{method}_" + results_model.columns
        results_model.to_csv(results_model_dir, index=False)

    if bagging:
        # Run with bagging strategy
        marker = f"{data_name}_{method}-bagging_num_train-{num_train}_rs-{random_state}"
        results_model_dir = result_save_dir_base + f"/{marker}.csv"
        if os.path.exists(results_model_dir):
            pass
        else:
            estimator_bagging.fit(X_train, y_train)
            if task == "regression":
                y_pred = estimator_bagging.predict(X_test)
            else:
                y_pred = estimator_bagging.predict_proba(X_test)
                y_pred = reshape_pred_output(y_pred)
            y_pred = check_pred_output(y_train, y_pred)
            score = return_score(y_test, y_pred, task)

            results_ = dict()
            results_[result_criterion[0]] = score[0]
            results_[result_criterion[1]] = score[1]
            results_model = pd.DataFrame([results_], columns=result_criterion[:2])
            results_model.columns = f"{method}_" + results_model.columns
            results_model.to_csv(results_model_dir, index=False)

    return None


def get_executor_slurm(
    job_name,
    timeout_hour=60,
    n_cpus=10,
    max_parallel_tasks=10,
    partition="parietal,normal",
    exclude="marg[037-038,042-044]",
    gpu=False,
):
    """Return a submitit executor to launch various tasks on a SLURM cluster.

    Parameters
    ----------
    job_name: str
        Name of the tasks that will be run. It will be used to create an output
        directory and display task info in squeue.
    timeout_hour: int
        Maximal number of hours the task will run before being interupted.
    n_cpus: int
        Number of CPUs requested for each task.
    max_parallel_tasks: int
        Maximal number of tasks that will run at once. This can be used to
        limit the total amount of the cluster used by a script.
    partition: str
        Partition of SLURM where the job would be submitted to.
    exclude: str
        Name of nodes to exclude to submitting jobs.
    gpu: bool
        If set to True, require one GPU per task.
    """

    executor = submitit.AutoExecutor(job_name)
    executor.update_parameters(
        timeout_min=180,
        slurm_job_name=job_name,
        slurm_time=f"{timeout_hour}:00:00",
        array_parallelism=max_parallel_tasks,
        slurm_additional_parameters={
            "ntasks": 1,
            "partition": f"{partition}",
            "exclude": f"{exclude}",
            "cpus-per-task": n_cpus,
            "distribution": "block:block",
        },
    )
    if gpu:
        executor.update_parameters(
            slurm_gres=f"gpu:1",
            slurm_setup=[
                "#SBATCH -C v100-32g",  # Require a specific resource
                "module purge",
                "module load cuda/10.1.2 "  # Load drivers
                "cudnn/7.6.5.32-cuda-10.1 nccl/2.5.6-2-cuda",
            ],
        )
    return executor


def _get_experiment_args_list(
    data_name,
    num_train,
    method,
    random_state,
    bagging,
    device,
):
    """Returns the list of arguments to run evaluations."""

    # Setting for train size
    if "all" in data_name:
        data_name_list = carte_datalist
    else:
        data_name_list = data_name
        if isinstance(data_name_list, list) == False:
            data_name_list = [data_name_list]

    # Setting for train size
    if "all" in num_train:
        num_train = [32, 64, 128, 256, 512, 1024, 2048]
    else:
        if isinstance(num_train, list) == False:
            num_train = [num_train]
            num_train = list(map(int, num_train))
        else:
            num_train = list(map(int, num_train))

    # Setting for bagging
    if bagging == "True":
        bagging = True
    else:
        bagging = False

    # Setting for methods
    if "full" in method:
        method_list = carte_singletable_baselines["full"]
    elif "reduced" in method:
        assert bagging == False
        method_list = carte_singletable_baselines["reduced"]
    elif "f-r" in method:
        method_list = set(carte_singletable_baselines["full"])
        method_list -= set(carte_singletable_baselines["reduced"])
        method_list = list(method_list)
        method_list.sort()
    else:
        method_list = method
        if isinstance(method_list, list) == False:
            method_list = [method_list]

    # Setting for random state
    if "all" in random_state:
        random_state = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    else:
        if isinstance(random_state, list) == False:
            random_state = [random_state]
            random_state = list(map(int, random_state))
        else:
            random_state = list(map(int, random_state))

    # List out all the cases and run
    args_dict = dict()
    args_dict["data_name"] = data_name_list
    args_dict["num_train"] = num_train
    args_dict["method"] = method_list
    args_dict["random_state"] = random_state
    args_dict["device"] = [device]
    args_dict["bagging"] = [bagging]
    args_list = list(ParameterGrid(args_dict))

    return args_list


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation in parallel on SLURM.")
    parser.add_argument(
        "-jn",
        "--job_name",
        type=str,
        help="Name of job submitted.",
    )
    parser.add_argument(
        "-t",
        "--timeout_hour",
        type=int,
        default=10,
        help="Number of CPUs per run of run_one.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Whether or not to run computation on a GPU.",
    )
    parser.add_argument(
        "-w",
        "--n-cpus",
        type=int,
        default=10,
        help="Number of CPUs per run of run_one.",
    )
    parser.add_argument(
        "-mpt",
        "--max_parallel_tasks",
        type=int,
        default=10,
        help="Maximal number of tasks that will run at once.",
    )
    parser.add_argument(
        "-p",
        "--partition",
        type=str,
        default="parietal,normal",
        help="Partition of SLURM to submits jobs to.",
    )
    parser.add_argument(
        "-ex",
        "--exclude",
        type=str,
        default="marg[037-038,042-044]",
        help="Nodes to exclude for submitting jobs.",
    )
    parser.add_argument(
        "-dn",
        "--data_name",
        nargs="+",
        type=str,
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "-nt",
        "--num_train",
        nargs="+",
        type=str,
        help="Number of train",
    )
    parser.add_argument(
        "-m",
        "--method",
        nargs="+",
        type=str,
        help="Method to evaluate",
    )
    parser.add_argument(
        "-rs",
        "--random_state",
        nargs="+",
        type=str,
        help="Random_state",
    )
    parser.add_argument(
        "-b",
        "--bagging",
        type=str,
        help="include bagging strategy for evaluation",
    )
    parser.add_argument(
        "-dv",
        "--device",
        type=str,
        help="Device, cpu or cuda",
    )
    args = parser.parse_args()

    # List all parameters to run the computation
    args_list = _get_experiment_args_list(
        args.data_name,
        args.num_train,
        args.method,
        args.random_state,
        args.bagging,
        args.device,
    )

    # Submit one task per set of parameters
    executor = get_executor_slurm(
        job_name=args.job_name,
        timeout_hour=args.timeout_hour,
        n_cpus=args.n_cpus,
        max_parallel_tasks=args.max_parallel_tasks,
        partition=args.partition,
        exclude=args.exclude,
    )

    # Run the computation on SLURM cluster with `submitit`
    print("Submitting jobs...", end="", flush=True)
    with executor.batch():
        tasks = [executor.submit(run_model, **args) for args in args_list]

    t_start = time.time()
    print("done")
