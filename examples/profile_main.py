# Import necessary packages
import os
import sys
import time
import psutil
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, roc_auc_score
import cProfile
import pstats
from line_profiler import LineProfiler

# Import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.carte_table_to_graph import Table2GraphTransformer
from src.carte_estimator import CARTERegressor, CARTEClassifier
from configs.directory import config_directory

# Initialize process for memory monitoring
process = psutil.Process()

def profile(func):
    """Decorator for line profiling individual functions."""
    def wrapper(*args, **kwargs):
        lp = LineProfiler()
        lp.add_function(func)
        result = lp(func)(*args, **kwargs)
        lp.print_stats()
        return result
    return wrapper

@profile
def load_data(data_name):
    """Load and preprocess the data."""
    data_pd_dir = f"{config_directory['data_singletable']}/{data_name}/raw.parquet"
    data_pd = pd.read_parquet(data_pd_dir)
    data_pd.fillna(value=np.nan, inplace=True)
    config_data_dir = f"{config_directory['data_singletable']}/{data_name}/config_data.json"
    with open(config_data_dir) as filename:
        config_data = json.load(filename)
    return data_pd, config_data

@profile
def set_train_test_split(data, data_config, num_train, random_state):
    """Set train/test split given the random state."""
    target_name = data_config["target_name"]
    X = data.drop(columns=target_name)
    y = np.array(data[target_name])

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

def main():
    # Set basic specifications
    data_name = "wina_pl"  # Name of the data
    num_train = 128        # Train-size
    random_state = 1       # Random_state
    num_cpu = 8            # Number of CPUs used

    # Start profiling with cProfile
    pr = cProfile.Profile()
    pr.enable()

    # Load data
    data, data_config = load_data(data_name)

    # Set train/test split
    X_train_, X_test_, y_train, y_test = set_train_test_split(
        data,
        data_config,
        num_train,
        random_state,
    )

    # Transform data into graphs
    preprocessor = Table2GraphTransformer()
    X_train = preprocessor.fit_transform(X_train_, y=y_train)
    X_test = preprocessor.transform(X_test_)

    # Define and fit the estimator for Regression
    estimator = CARTERegressor(num_model=10, disable_pbar=False, random_state=0, device='cpu', n_jobs=10)
    estimator.fit(X=X_train, y=y_train)

    # Predict and compute R2 score
    y_pred = estimator.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"\nThe R2 score for CARTE: {r2:.4f}")

    # Stop profiling and print stats
    pr.disable()
    ps = pstats.Stats(pr)
    ps.sort_stats(pstats.SortKey.TIME).print_stats(10)

if __name__ == "__main__":
    main()
