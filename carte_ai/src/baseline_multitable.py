"""Baselines for multitable problem."""

import pandas as pd
import numpy as np

from typing import Union
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_random_state
from sklearn.metrics import r2_score, roc_auc_score
from joblib import Parallel, delayed

from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)


class GradientBoostingMultitableBase(BaseEstimator):
    """Base class for Gradient Boosting Multitable Estimator."""

    def __init__(
        self,
        *,
        source_data,
        source_fraction,
        num_model,
        val_size,
        random_state,
        n_jobs,
    ):
        self.source_data = source_data
        self.source_fraction = source_fraction
        self.num_model = num_model
        self.val_size = val_size
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit the model.

        Parameters
        ----------
        X : Pandas dataframe of the target dataset (n_samples)
            The input samples.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
               Fitted estimator.
        """

        # Preliminary settings
        self.is_fitted_ = False
        self.X_ = X
        self.y_ = y
        self._set_gb_method()

        # Set random_state
        random_state = check_random_state(self.random_state)
        random_state_list = [random_state.randint(10000) for _ in range(self.num_model)]

        # Run parallel for different train/validation split
        result_fit = Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_fit_with_source_split)(X, y, rs)
            for rs in random_state_list
        )

        # Store the required results that may be used later
        self.estimator_list_ = [model for (model, _) in result_fit]
        self.valid_loss_ = [valid_loss for (_, valid_loss) in result_fit]

        self.is_fitted_ = True

        return self

    def _run_fit_with_source_split(self, X, y, random_state):
        """Train each model corresponding to the random_state with the split on Source and train/validtion on Target.

        Returns the trained estimator, and the validation loss of the train model.
        """

        # Set validation by val_size
        stratify = None
        if self._estimator_type == "classifier":
            stratify = self.y_
        dx_train, dx_valid, dy_train, dy_valid = train_test_split(
            X,
            y,
            test_size=self.val_size,
            shuffle=True,
            stratify=stratify,
            random_state=random_state,
        )

        # Set source data
        X_train_source, y_train_source = self._load_source_data(random_state)

        # Total dataset
        X_train = pd.concat([dx_train, X_train_source], axis=0)
        y_train = pd.concat([dy_train, y_train_source], axis=0)

        # Set estimator, run fit/predict to obtain validation loss
        estimator = self._set_estimator()
        estimator.fit(X_train, y_train)
        y_pred = self._generate_output(estimator, dx_valid)
        valid_loss = self._return_score(dy_valid, y_pred)

        return estimator, valid_loss

    def _load_source_data(self, random_state):
        """Loads the Source data and extract based on the defined fraction of Source.

        Applies stratification on the Source data based on their sizes.
        The max. size of the source data is set at 10,000 to prevent overfitting on the Source.
        """
        # Set train_size (max = 10000)
        if len(self.source_data["X"]) > 10000:
            train_size = 10000 / len(self.source_data["X"]) * self.source_fraction
        else:
            train_size = self.source_fraction
        # Set split for source data
        if self._estimator_type == "regressor":
            stratify = self.source_data["domain_indicator"]
        if self._estimator_type == "classifier":
            y_source_temp = self.source_data["y"].copy()
            y_source_temp = y_source_temp.astype(str)
            stratify = self.source_data["domain_indicator"] + "_" + y_source_temp
        X_train_source, _, y_train_source, _ = train_test_split(
            self.source_data["X"],
            self.source_data["y"],
            train_size=train_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify,
        )
        return X_train_source, y_train_source

    def _generate_output(self, estimator, X):
        """Generate output on the given estimator and X."""

        # Predict
        if self._estimator_type == "regressor":
            y_pred = estimator.predict(X)
        else:
            y_pred = estimator.predict_proba(X)
        # Reshape prediction
        if self._estimator_type == "classifier":
            num_pred = len(y_pred)
            if y_pred.shape == (num_pred, 2):
                y_pred = y_pred[:, 1]
            elif y_pred.shape == (num_pred, 1):
                y_pred = y_pred.ravel()
            else:
                pass
        # Control for nan in prediction
        if np.isnan(y_pred).sum() > 0:
            mean_pred = np.mean(self.y_)
            y_pred[np.isnan(y_pred)] = mean_pred
        return y_pred

    def _return_score(self, y, y_pred):
        """Return the score based on the task."""
        if self._estimator_type == "regressor":
            score = r2_score(y, y_pred)
        else:
            score = roc_auc_score(y, y_pred)
        return score

    def _set_estimator(self):
        """Set the estimator according to the model of Gradient-Boosted Trees."""

        fixed_params = dict()
        if self.gb_method_ == "catboost":
            fixed_params["cat_features"] = self.cat_features_
            fixed_params["verbose"] = False
            fixed_params["allow_writing_files"] = False
            fixed_params["thread_count"] = self.thread_count
            fixed_params["max_ctr_complexity"] = 2
            catboost_params = dict()
            catboost_params["max_depth"] = self.max_depth
            catboost_params["learning_rate"] = self.learning_rate
            catboost_params["bagging_temperature"] = self.bagging_temperature
            catboost_params["l2_leaf_reg"] = self.l2_leaf_reg
            catboost_params["one_hot_max_size"] = self.one_hot_max_size
            catboost_params["iterations"] = self.iterations
            if self._estimator_type == "regressor":
                estimator_ = CatBoostRegressor(**fixed_params, **catboost_params)
            else:
                estimator_ = CatBoostClassifier(**fixed_params, **catboost_params)
        elif self.gb_method_ == "xgboost":
            fixed_params["booster"] = "gbtree"
            fixed_params["tree_method"] = "exact"  # exact approx hist
            xgb_params = dict()
            xgb_params["n_estimators"] = self.n_estimators
            xgb_params["max_depth"] = self.max_depth
            xgb_params["min_child_weight"] = self.min_child_weight
            xgb_params["subsample"] = self.subsample
            xgb_params["learning_rate"] = self.learning_rate
            xgb_params["colsample_bylevel"] = self.colsample_bylevel
            xgb_params["colsample_bytree"] = self.colsample_bytree
            xgb_params["gamma"] = self.reg_gamma
            xgb_params["lambda"] = self.reg_lambda
            xgb_params["alpha"] = self.reg_alpha
            if self._estimator_type == "regressor":
                estimator_ = XGBRegressor(**fixed_params, **xgb_params)
            else:
                estimator_ = XGBClassifier(**fixed_params, **xgb_params)
        elif self.gb_method_ == "histgb":
            histgb_params = dict()
            histgb_params["learning_rate"] = self.learning_rate
            histgb_params["max_depth"] = self.max_depth
            histgb_params["max_leaf_nodes"] = self.max_leaf_nodes
            histgb_params["min_samples_leaf"] = self.min_samples_leaf
            histgb_params["l2_regularization"] = self.l2_regularization
            if self._estimator_type == "regressor":
                estimator_ = HistGradientBoostingRegressor(
                    **fixed_params, **histgb_params
                )
            else:
                estimator_ = HistGradientBoostingClassifier(
                    **fixed_params, **histgb_params
                )
        return estimator_

    def _set_gb_method(
        self,
    ):
        self.gb_method_ = None
        return None


class GradientBoostingRegressorBase(RegressorMixin, GradientBoostingMultitableBase):
    """Base class for Gradient Boosting Multitable Regressor."""

    def __init__(
        self,
        *,
        source_data,
        source_fraction,
        num_model,
        val_size,
        random_state,
        n_jobs,
    ):
        super(GradientBoostingRegressorBase, self).__init__(
            source_data=source_data,
            source_fraction=source_fraction,
            num_model=num_model,
            val_size=val_size,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def predict(self, X):
        """Predict values for X. Returns the average of predicted values over all the models.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self, "is_fitted_")
        # Obtain output
        X_test = X.copy()
        out = [estimator.predict(X_test) for estimator in self.estimator_list_]
        if self.num_model == 1:
            out = np.array(out).squeeze().transpose()
        else:
            out = np.array(out).squeeze().transpose()
            out = np.mean(out, axis=1)
        # Control for nan in prediction
        if np.isnan(out).sum() > 0:
            mean_pred = np.mean(self.y_)
            out[np.isnan(out)] = mean_pred
        return out


class GradientBoostingClassifierBase(ClassifierMixin, GradientBoostingMultitableBase):
    """Base class for Gradient Boosting Multitable Classifier."""

    def __init__(
        self,
        *,
        source_data,
        source_fraction,
        num_model,
        val_size,
        random_state,
        n_jobs,
    ):
        super(GradientBoostingClassifierBase, self).__init__(
            source_data=source_data,
            source_fraction=source_fraction,
            num_model=num_model,
            val_size=val_size,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def predict(self, X):
        """Predict classes for X.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self, "is_fitted_")
        return np.round(self.predict_proba(X))

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
            The input samples.

        Returns
        -------
        p : ndarray, shape (n_samples,) for binary classification or (n_samples, n_classes)
            The class probabilities of the input samples.
        """

        check_is_fitted(self, "is_fitted_")
        # Obtain output
        out = [estimator.predict_proba(X)[:, 1] for estimator in self.estimator_list_]
        if self.num_model == 1:
            out = np.array(out).transpose()
        else:
            out = np.array(out).squeeze().transpose()
            out = np.mean(out, axis=1)
        # Control for nan in prediction
        if np.isnan(out).sum() > 0:
            mean_pred = np.mean(self.y_)
            out[np.isnan(out)] = mean_pred
        return out

    def decision_function(self, X):
        """Compute the decision function of X."""
        decision = self.predict_proba(X)
        return decision


class CatBoostMultitableRegressor(GradientBoostingRegressorBase):
    """Base class for CatBoost Multitable Regressor."""

    def __init__(
        self,
        *,
        source_data: dict = {},
        max_depth: int = 6,
        learning_rate: float = 0.03,
        bagging_temperature: float = 1,
        l2_leaf_reg: float = 3.0,
        one_hot_max_size: int = 2,
        iterations: int = 1000,
        thread_count: int = 1,
        source_fraction: float = 0.5,
        num_model: int = 1,
        val_size: float = 0.1,
        random_state: int = 0,
        n_jobs: int = 1,
    ):
        super(CatBoostMultitableRegressor, self).__init__(
            source_data=source_data,
            source_fraction=source_fraction,
            num_model=num_model,
            val_size=val_size,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.bagging_temperature = bagging_temperature
        self.l2_leaf_reg = l2_leaf_reg
        self.one_hot_max_size = one_hot_max_size
        self.iterations = iterations
        self.thread_count = thread_count

    def _set_gb_method(
        self,
    ):
        """Set the Gradient-Boosting method.

        For CatBoost, it sets the required indicators of categorical columns.
        """
        self.gb_method_ = "catboost"
        # Set column names
        X_total_train = pd.concat([self.X_, self.source_data["X"]], axis=0)
        self.cat_col_names_ = X_total_train.select_dtypes(
            include="object"
        ).columns.tolist()
        self.cat_features_ = [
            X_total_train.columns.get_loc(col) for col in self.cat_col_names_
        ]
        return None


class CatBoostMultitableClassifier(GradientBoostingClassifierBase):
    """Base class for CatBoost Multitable Classifier."""

    def __init__(
        self,
        *,
        source_data: dict = {},
        max_depth: int = 6,
        learning_rate: float = 0.03,
        bagging_temperature: float = 1,
        l2_leaf_reg: float = 3.0,
        one_hot_max_size: int = 2,
        iterations: int = 1000,
        thread_count: int = 1,
        source_fraction: float = 0.5,
        num_model: int = 1,
        val_size: float = 0.1,
        random_state: int = 0,
        n_jobs: int = 1,
    ):
        super(CatBoostMultitableClassifier, self).__init__(
            source_data=source_data,
            source_fraction=source_fraction,
            num_model=num_model,
            val_size=val_size,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.bagging_temperature = bagging_temperature
        self.l2_leaf_reg = l2_leaf_reg
        self.one_hot_max_size = one_hot_max_size
        self.iterations = iterations
        self.thread_count = thread_count

    def _set_gb_method(
        self,
    ):
        """Set the Gradient-Boosting method.

        For CatBoost, it sets the required indicators of categorical columns.
        """
        self.gb_method_ = "catboost"
        # Set column names
        X_total_train = pd.concat([self.X_, self.source_data["X"]], axis=0)
        self.cat_col_names_ = X_total_train.select_dtypes(
            include="object"
        ).columns.tolist()
        self.cat_features_ = [
            X_total_train.columns.get_loc(col) for col in self.cat_col_names_
        ]
        return None


class HistGBMultitableRegressor(GradientBoostingRegressorBase):
    """Base class for Historgram Gradient Boosting Multitable Regressor."""

    def __init__(
        self,
        *,
        source_data: dict = {},
        learning_rate: float = 0.1,
        max_depth: Union[None, int] = None,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0,
        source_fraction: float = 0.5,
        num_model: int = 1,
        val_size: float = 0.1,
        random_state: int = 0,
        n_jobs: int = 1,
    ):
        super(HistGBMultitableRegressor, self).__init__(
            source_data=source_data,
            source_fraction=source_fraction,
            num_model=num_model,
            val_size=val_size,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization

    def _set_gb_method(
        self,
    ):
        """Set the Gradient-Boosting method."""
        self.gb_method_ = "histgb"
        return None


class HistGBMultitableClassifier(GradientBoostingClassifierBase):
    """Base class for Historgram Gradient Boosting Multitable Classifier."""

    def __init__(
        self,
        *,
        source_data: dict = {},
        learning_rate: float = 0.1,
        max_depth: Union[None, int] = None,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0,
        source_fraction: float = 0.5,
        num_model: int = 1,
        val_size: float = 0.1,
        random_state: int = 0,
        n_jobs: int = 1,
    ):
        super(HistGBMultitableClassifier, self).__init__(
            source_data=source_data,
            source_fraction=source_fraction,
            num_model=num_model,
            val_size=val_size,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization

    def _set_gb_method(
        self,
    ):
        """Set the Gradient-Boosting method."""
        self.gb_method_ = "histgb"
        return None


class XGBoostMultitableRegressor(GradientBoostingRegressorBase):
    """Base class for XGBoost Multitable Regressor."""

    def __init__(
        self,
        *,
        source_data: dict = {},
        n_estimators: int = 100,
        max_depth: int = 6,
        min_child_weight: float = 1,
        subsample: float = 1,
        learning_rate: float = 0.3,
        colsample_bylevel: float = 1,
        colsample_bytree: float = 1,
        reg_gamma: float = 0,
        reg_lambda: float = 1,
        reg_alpha: float = 0,
        source_fraction: float = 0.5,
        num_model: int = 1,
        val_size: float = 0.1,
        random_state: int = 0,
        n_jobs: int = 1,
    ):
        super(XGBoostMultitableRegressor, self).__init__(
            source_data=source_data,
            source_fraction=source_fraction,
            num_model=num_model,
            val_size=val_size,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.learning_rate = learning_rate
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bytree = colsample_bytree
        self.reg_gamma = reg_gamma
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha

    def _set_gb_method(
        self,
    ):
        """Set the Gradient-Boosting method."""
        self.gb_method_ = "xgboost"
        return None


class XGBoostMultitableClassifier(GradientBoostingClassifierBase):
    """Base class for XGBoost Multitable Classifier."""

    def __init__(
        self,
        *,
        source_data: dict = {},
        n_estimators: int = 100,
        max_depth: int = 6,
        min_child_weight: float = 1,
        subsample: float = 1,
        learning_rate: float = 0.3,
        colsample_bylevel: float = 1,
        colsample_bytree: float = 1,
        reg_gamma: float = 0,
        reg_lambda: float = 1,
        reg_alpha: float = 0,
        source_fraction: float = 0.5,
        num_model: int = 1,
        val_size: float = 0.1,
        random_state: int = 0,
        n_jobs: int = 1,
    ):
        super(XGBoostMultitableClassifier, self).__init__(
            source_data=source_data,
            source_fraction=source_fraction,
            num_model=num_model,
            val_size=val_size,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.learning_rate = learning_rate
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bytree = colsample_bytree
        self.reg_gamma = reg_gamma
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha

    def _set_gb_method(
        self,
    ):
        """Set the Gradient-Boosting method."""
        self.gb_method_ = "xgboost"
        return None
