"""Custom grid search used for CARTE-GNN model"""

import ast
import copy
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from time import perf_counter
from sklearn.model_selection import ParameterGrid


def carte_gridsearch(
    estimator,
    X_train: list,
    y_train: np.array,
    param_distributions: dict,
    refit: bool = True,
    n_jobs: int = 1,
):
    """CARTE grid search.

    This function runs grid search for CARTE GNN models.

    Parameters
    ----------
    estimator : CARTE estimator
        The CARTE estimator used for grid search
    X_train : list
        The list of graph objects for the train data transformed using Table2GraphTransformer
    y_train : numpy array of shape (n_samples,)
        The target variable of the train data.
    param_distributions: dict
        The dictionary of parameter grids to search for the optimial parameter.
    refit: bool, default=True
        Indicates whether to return a refitted estimator with the best parameter.
    n_jobs: int, default=1
        Number of jobs to run in parallel. Training the estimator in the grid search is parallelized
        over the parameter grid.

    Returns
    -------
    Result : Pandas DataFrame
        The result of each parameter grid.
    best_params : dict
        The dictionary of best parameters obtained through grid search.
    best_estimator : CARTEGNN estimator
        The CARTE estimator trained using the best_params if refit is set to True.
    """
    # Set paramater list
    param_distributions_ = param_distributions.copy()
    param_list = list(ParameterGrid(param_distributions_))

    # Run Gridsearch
    gridsearch_result = Parallel(n_jobs=n_jobs)(
        delayed(_run_search_carte)(estimator, X_train, y_train, params)
        for params in param_list
    )
    gridsearch_result = pd.concat(gridsearch_result, axis=0)

    # Add rank
    rank = gridsearch_result["score"].rank(method="min").astype(int).copy()
    rank = pd.DataFrame(rank)
    rank.rename(columns={"score": "rank"}, inplace=True)
    gridsearch_result = pd.concat([gridsearch_result, rank], axis=1)

    # Best params
    params_ = gridsearch_result["params"]
    best_params_ = params_[gridsearch_result["rank"] == 1].iloc[0]
    best_params = ast.literal_eval(best_params_)

    # Refit
    best_estimator = None
    if refit:
        best_estimator = copy.deepcopy(estimator)
        best_estimator.__dict__.update(best_params)
        best_estimator.fit(X=X_train, y=y_train)

    return gridsearch_result, best_params, best_estimator


def _run_search_carte(estimator, X_train, y_train, params):
    """Run fit predict over a parmeter in the parameter grid."""
    # Measure time
    start_time = perf_counter()

    # Run estimator
    estimator_ = copy.deepcopy(estimator)
    estimator_.__dict__.update(params)
    estimator_.fit(X=X_train, y=y_train)

    # Measure time
    end_time = perf_counter()
    duration = round(end_time - start_time, 4)

    # Statistics
    vl = np.array(estimator_.valid_loss_)

    # Obtain results
    result_run = {
        f"cv-run_{i}_valid_loss": estimator_.valid_loss_[i]
        for i in range(estimator_.num_model)
    }
    result_run["params"] = str(params)
    result_run["score"] = np.mean(vl)
    result_run["fit_time"] = duration

    result_df = pd.DataFrame([result_run])
    result_df = result_df.reindex(sorted(result_df.columns), axis=1)

    return result_df
