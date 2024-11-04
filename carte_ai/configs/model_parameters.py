"""
Parameter distributions for hyperparameter optimization
"""

from scipy.stats import loguniform, randint, uniform, norm
import copy


class loguniform_int:
    """Integer valued version of the log-uniform distribution"""

    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)


class norm_int:
    """Integer valued version of the normal distribution"""

    def __init__(self, a, b):
        self._distribution = norm(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        if self._distribution.rvs(*args, **kwargs).astype(int) < 1:
            return 1
        else:
            return self._distribution.rvs(*args, **kwargs).astype(int)


param_distributions_total = dict()

# carte-gnn
param_distributions = dict()
lr_grid = [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3]
param_distributions["learning_rate"] = lr_grid
param_distributions_total["carte-gnn"] = param_distributions

# histgb
param_distributions = dict()
param_distributions["learning_rate"] = loguniform(1e-2, 10)
param_distributions["max_depth"] = [None, 2, 3, 4]
param_distributions["max_leaf_nodes"] = norm_int(31, 5)
param_distributions["min_samples_leaf"] = norm_int(20, 2)
param_distributions["l2_regularization"] = loguniform(1e-6, 1e3)
param_distributions_total["histgb"] = param_distributions

# catboost
param_distributions = dict()
param_distributions["max_depth"] = randint(2, 11)
param_distributions["learning_rate"] = loguniform(1e-5, 1)
param_distributions["bagging_temperature"] = uniform(0, 1)
param_distributions["l2_leaf_reg"] = loguniform(1, 10)
param_distributions["iterations"] = randint(400, 1001)
param_distributions["one_hot_max_size"] = randint(2, 26)
param_distributions_total["catboost"] = param_distributions

# xgb
param_distributions = dict()
param_distributions["n_estimators"] = randint(50, 1001)
param_distributions["max_depth"] = randint(2, 11)
param_distributions["min_child_weight"] = loguniform(1, 100)
param_distributions["subsample"] = uniform(0.5, 1 - 0.5)
param_distributions["learning_rate"] = loguniform(1e-5, 1)
param_distributions["colsample_bylevel"] = uniform(0.5, 1 - 0.5)
param_distributions["colsample_bytree"] = uniform(0.5, 1 - 0.5)
param_distributions["gamma"] = loguniform(1e-8, 7)
param_distributions["lambda"] = loguniform(1, 4)
param_distributions["alpha"] = loguniform(1e-8, 100)
param_distributions_total["xgb"] = param_distributions

# RandomForest
param_distributions = dict()
param_distributions["n_estimators"] = randint(50, 250)
param_distributions["max_depth"] = [None, 2, 3, 4]
param_distributions["max_features"] = [
    "sqrt",
    "log2",
    None,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]
param_distributions["min_samples_leaf"] = loguniform_int(0.5, 50.5)
param_distributions["bootstrap"] = [True, False]
param_distributions["min_impurity_decrease"] = [0.0, 0.01, 0.02, 0.05]
param_distributions_total["randomforest"] = param_distributions


# resnet
param_distributions = dict()
param_distributions["normalization"] = ["batchnorm", "layernorm"]
param_distributions["num_layers"] = randint(1, 9)
param_distributions["hidden_dim"] = randint(32, 513)
param_distributions["hidden_factor"] = randint(1, 3)
param_distributions["hidden_dropout_prob"] = uniform(0.0, 0.5)
param_distributions["residual_dropout_prob"] = uniform(0.0, 0.5)
param_distributions["learning_rate"] = loguniform(1e-5, 1e-2)
param_distributions["weight_decay"] = loguniform(1e-8, 1e-2)
param_distributions["batch_size"] = [16, 32]
param_distributions_total["resnet"] = param_distributions

# mlp
param_distributions = dict()
param_distributions["hidden_dim"] = [2**x for x in range(4, 11)]
param_distributions["num_layers"] = randint(1, 5)
param_distributions["dropout_prob"] = uniform(0.0, 0.5)
param_distributions["learning_rate"] = loguniform(1e-5, 1e-2)
param_distributions["weight_decay"] = loguniform(1e-8, 1e-2)
param_distributions["batch_size"] = [16, 32]
param_distributions_total["mlp"] = param_distributions

# ridge regression
param_distributions = dict()
param_distributions["solver"] = ["svd", "cholesky", "lsqr", "sag"]
param_distributions["alpha"] = loguniform(1e-5, 100)
param_distributions_total["ridge"] = param_distributions

# logistic regression
param_distributions = dict()
param_distributions["solver"] = ["newton-cg", "lbfgs", "liblinear"]
param_distributions["penalty"] = ["none", "l1", "l2", "elasticnet"]
param_distributions["C"] = loguniform(1e-5, 100)
param_distributions_total["logistic"] = param_distributions

# tabpfn
param_distributions = dict()
param_distributions_total["tabpfn"] = param_distributions

# catboost-multitable
param_distributions = copy.deepcopy(param_distributions_total["catboost"])
param_distributions["source_fraction"] = uniform(0, 1)
param_distributions_total["catboost-multitable"] = param_distributions

# histgb-multitable
param_distributions = copy.deepcopy(param_distributions_total["histgb"])
param_distributions["source_fraction"] = uniform(0, 1)
param_distributions_total["histgb-multitable"] = param_distributions
