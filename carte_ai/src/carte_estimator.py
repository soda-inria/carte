"""CARTE estimators for regression and classification."""

import torch
import torch.cuda.amp as amp
import numpy as np
import pandas as pd
import copy
import math
from typing import Union
from torcheval.metrics import (
    MeanSquaredError,
    R2Score,
    BinaryAUROC,
    BinaryNormalizedEntropy,
    BinaryAUPRC,
    MulticlassAUROC,
)
from torch import Tensor
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedShuffleSplit,
    ParameterGrid,
    train_test_split,
)
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_random_state
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.special import softmax
from carte_ai.src.carte_model import CARTE_NN_Model, CARTE_NN_Model_Ablation
from carte_ai.configs.directory import config_directory


class BaseCARTEEstimator(BaseEstimator):
    """Base class for CARTE Estimator."""

    def __init__(
        self,
        *,
        num_layers,
        load_pretrain,
        freeze_pretrain,
        learning_rate,
        batch_size,
        max_epoch,
        dropout,
        val_size,
        cross_validate,
        early_stopping_patience,
        num_model,
        random_state,
        n_jobs,
        device,
        disable_pbar,
        pretrained_model_path=config_directory["pretrained_model"],
    ):
        self.num_layers = num_layers
        self.load_pretrain = load_pretrain
        self.freeze_pretrain = freeze_pretrain
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.dropout = dropout
        self.val_size = val_size
        self.cross_validate = cross_validate
        self.early_stopping_patience = early_stopping_patience
        self.num_model = num_model
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.device = device
        self.disable_pbar = disable_pbar
        self.pretrained_model_path = pretrained_model_path

    def fit(self, X, y):
        """Fit the CARTE model.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
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
        self.device_ = torch.device(self.device)
        self.X_ = X
        self.y_ = y
        self._set_task_specific_settings()

        # Set the cv-splits
        splits = self._set_train_valid_split()

        # Fit model
        result_fit = Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_train_with_early_stopping)(X, split_index)
            for split_index in splits
        )

        # Store the required results that may be used later
        self.model_list_ = [model for (model, _) in result_fit]
        self.valid_loss_ = [valid_loss for (_, valid_loss) in result_fit]
        self.weights_ = np.array([1 / self.num_model] * self.num_model)
        self.is_fitted_ = True

        return self

    def _run_train_with_early_stopping(self, X, split_index):
        """Train each model corresponding to the random_state with the early_stopping patience.

        This mode of training sets train/valid set for the early stopping criterion.
        Returns the trained model, and the validation loss at the best epoch.
        """

        # Set datasets
        ds_train = [X[i] for i in split_index[0]]
        ds_valid = [X[i] for i in split_index[1]]

        # Set validation batch for evaluation
        ds_valid_eval = self._set_data_eval(data=ds_valid)

        # Load model and optimizer
        model_run_train = self._load_model()
        model_run_train.to(self.device_)
        optimizer = torch.optim.AdamW(
            model_run_train.parameters(), lr=self.learning_rate
        )

        # Initialize GradScaler
        scaler = amp.GradScaler()

        # Train model
        train_loader = DataLoader(ds_train, batch_size=self.batch_size, shuffle=False)
        valid_loss_best = 9e15
        es_counter = 0
        model_best_ = copy.deepcopy(model_run_train)
        for _ in tqdm(
            range(1, self.max_epoch + 1),
            desc=f"Model No. xx",
            disable=self.disable_pbar,
        ):
            self._run_epoch(model_run_train, optimizer, train_loader, scaler)
            valid_loss = self._eval(model_run_train, ds_valid_eval)
            if valid_loss < valid_loss_best:
                valid_loss_best = valid_loss
                model_best_ = copy.deepcopy(model_run_train)
                es_counter = 0
            else:
                es_counter += 1
                if es_counter > self.early_stopping_patience:
                    break
        model_best_.eval()
        return model_best_, valid_loss_best

    def _run_epoch(self, model, optimizer, train_loader, scaler):
        """Run an epoch of the input model.

        Each epoch consists of steps that update the model and the optimizer.
        """
        model.train()
        for data in train_loader:  # Iterate in batches over the training dataset.
            self._run_step(model, data, optimizer, scaler)

    def _run_step(self, model, data, optimizer, scaler):
        """Run a step of the training.

        With each step, it updates the model and the optimizer.
        """
        optimizer.zero_grad()  # Clear gradients.
        data.to(self.device_)  # Send to device

        with amp.autocast():  # Enable autocasting
            out = model(data)  # Perform a single forward pass.
            target = data.y  # Set target
            if self.loss == "categorical_crossentropy":
                target = target.to(torch.long)
            if self.output_dim_ == 1:
                out = out.view(-1).to(torch.float32)  # Reshape output
                target = target.to(torch.float32)  # Reshape target
            loss = self.criterion_(out, target)  # Compute the loss.

        scaler.scale(loss).backward()  # Scale the loss and backward pass
        scaler.step(optimizer)  # Update parameters
        scaler.update()  # Update the scaler for next iteration

    def _eval(self, model, ds_eval):
        """Run an evaluation of the input data on the input model.

        Returns the selected loss of the input data from the input model.
        """
        with torch.no_grad():
            model.eval()
            with amp.autocast():  # Enable autocasting
                out = model(ds_eval)
                target = ds_eval.y
                if self.loss == "categorical_crossentropy":
                    target = target.to(torch.long)
                if self.output_dim_ == 1:
                    out = out.view(-1).to(torch.float32)
                    target = target.to(torch.float32)
                self.valid_loss_metric_.update(out, target)
                loss_eval = self.valid_loss_metric_.compute()
                loss_eval = loss_eval.detach().item()
                if self.valid_loss_flag_ == "neg":
                    loss_eval = -1 * loss_eval
                self.valid_loss_metric_.reset()
        return loss_eval

    def _set_train_valid_split(self):
        """Train/validation split for the bagging strategy.

        The style of split depends on the cross_validate parameter.
        Returns the train/validation split with KFold cross-validation.
        """

        if self._estimator_type == "regressor":
            if self.cross_validate:
                n_splits = int(1 / self.val_size)
                n_repeats = int(self.num_model / n_splits)
                splitter = RepeatedKFold(
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    random_state=self.random_state,
                )
            else:
                splitter = ShuffleSplit(
                    n_splits=self.num_model,
                    test_size=self.val_size,
                    random_state=self.random_state,
                )
            splits = [
                (train_index, test_index)
                for train_index, test_index in splitter.split(
                    np.arange(0, len(self.X_))
                )
            ]
        else:
            if self.cross_validate:
                n_splits = int(1 / self.val_size)
                n_repeats = int(self.num_model / n_splits)
                splitter = RepeatedStratifiedKFold(
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    random_state=self.random_state,
                )
            else:
                splitter = StratifiedShuffleSplit(
                    n_splits=self.num_model,
                    test_size=self.val_size,
                    random_state=self.random_state,
                )
            splits = [
                (train_index, test_index)
                for train_index, test_index in splitter.split(
                    np.arange(0, len(self.X_)), self.y_
                )
            ]

        return splits

    def _set_data_eval(self, data):
        """Constructs the aggregated graph object from the list of data.

        This is consistent with the graph object from torch_geometric.
        Returns the aggregated graph object.
        """
        make_batch = Batch()
        with torch.no_grad():
            ds_eval = make_batch.from_data_list(data, follow_batch=["edge_index"])
            ds_eval.to(self.device_)
        return ds_eval

    def _generate_output(self, X, model_list, weights):
        """Generate the output from the trained model.

        Returns the output (prediction) of input X.
        """

        # Obtain the batch to feed into the network
        ds_predict_eval = self._set_data_eval(data=X)
        with torch.no_grad():
            out = [
                model(ds_predict_eval).cpu().detach().numpy() for model in model_list
            ]
        out = np.average(out, weights=weights, axis=0)

        # Change if the task is classification
        if self.loss == "binary_crossentropy":
            out = 1 / (1 + np.exp(-out))
        elif self.loss == "categorical_crossentropy":
            out = softmax(out, axis=1)

        # Control for nulls in prediction
        if np.isnan(out).sum() > 0:
            mean_pred = np.mean(self.y_)
            out[np.isnan(out)] = mean_pred

        if out.ndim == 2 and out.shape[1] == 1:
            out = out.squeeze(axis=1)  # we don't want to squeeze first axis

        return out

    def _set_task_specific_settings(self):
        """Set task specific settings for regression and classfication."""

        if self._estimator_type == "regressor":
            if self.loss == "squared_error":
                self.criterion_ = torch.nn.MSELoss()
            elif self.loss == "absolute_error":
                self.criterion_ = torch.nn.L1Loss()
            if self.scoring == "squared_error":
                self.valid_loss_metric_ = MeanSquaredError()
                self.valid_loss_flag_ = "pos"
            elif self.scoring == "r2_score":
                self.valid_loss_metric_ = R2Score()
                self.valid_loss_flag_ = "neg"
            self.output_dim_ = 1
        elif self._estimator_type == "classifier":
            self.classes_ = np.unique(self.y_)
            if self.loss == "binary_crossentropy":
                self.criterion_ = torch.nn.BCEWithLogitsLoss()
                self.output_dim_ = 1
                if self.scoring == "auroc":
                    self.valid_loss_metric_ = BinaryAUROC()
                    self.valid_loss_flag_ = "neg"
                elif self.scoring == "binary_entropy":
                    self.valid_loss_metric_ = BinaryNormalizedEntropy(from_logits=True)
                    self.valid_loss_flag_ = "neg"
                elif self.scoring == "auprc":
                    self.valid_loss_metric_ = BinaryAUPRC()
                    self.valid_loss_flag_ = "neg"
            elif self.loss == "categorical_crossentropy":
                self.criterion_ = torch.nn.CrossEntropyLoss()
                self.output_dim_ = len(np.unique(self.y_))
                self.valid_loss_metric_ = MulticlassAUROC(num_classes=self.output_dim_)
                self.valid_loss_flag_ = "neg"
        self.valid_loss_metric_.to(self.device_)


    def _load_model(self):
        """Load the CARTE model for training.

        This loads the pretrained weights if the parameter load_pretrain is set to True.
        The freeze of the pretrained weights are controlled by the freeze_pretrain parameter.

        Returns the model that can be used for training.
        """
        # Model configuration
        model_config = dict()
        model_config["input_dim_x"] = self.X_[0].x.size(1)
        model_config["input_dim_e"] = self.X_[0].x.size(1)
        model_config["hidden_dim"] = self.X_[0].x.size(1)
        model_config["ff_dim"] = self.X_[0].x.size(1)
        model_config["num_heads"] = 12
        model_config["num_layers"] = self.num_layers - 1
        model_config["output_dim"] = self.output_dim_
        model_config["dropout"] = self.dropout

        # Set seed for torch - for reproducibility
        random_state = check_random_state(self.random_state)
        model_seed = random_state.randint(10000)
        torch.manual_seed(model_seed)

        # Set model architecture
        model = CARTE_NN_Model(**model_config)

        # Load the pretrained weights if specified
        if self.load_pretrain and self.pretrained_model_path:
            # Load the pretrained model from the specified path
            pretrain_model_dict = torch.load(
                self.pretrained_model_path, map_location=self.device_
            )

            # Rename the keys containing "initial_x" to "initial_x_pretrain"
            initial_x_keys = [
                key for key in pretrain_model_dict.keys() if "initial_x" in key
            ]
            for key in initial_x_keys:
                pretrain_model_dict[key + "_pretrain"] = pretrain_model_dict.pop(key)

            # Load the state dict into the model
            model.load_state_dict(pretrain_model_dict, strict=False)

        # Freeze the pretrained weights if specified
        if self.freeze_pretrain:
            for param in model.ft_base.read_out_block.parameters():
                param.requires_grad = False
            for param in model.ft_base.layers.parameters():
                param.requires_grad = False

        return model


class CARTERegressor(RegressorMixin, BaseCARTEEstimator):
    """CARTE Regressor for Regression tasks.

    This estimator is GNN-based model compatible with the CARTE pretrained model.

    Parameters
    ----------
    loss : {'squared_error', 'absolute_error'}, default='squared_error'
        The loss function used for backpropagation.
    scoring : {'r2_score', 'squared_error'}, default='r2_score'
        The scoring function used for validation.
    num_layers : int, default=1
        The number of layers for the NN model
    load_pretrain : bool, default=True
        Indicates whether to load pretrained weights or not
    freeze_pretrain : bool, default=True
        Indicates whether to freeze the pretrained weights in the training or not
    learning_rate : float, default=1e-3
        The learning rate of the model. The model uses AdamW as the optimizer
    batch_size : int, default=16
        The batch size used for training
    max_epoch : int or None, default=500
        The maximum number of epoch for training
    dropout : float, default=0
        The dropout rate for training
    val_size : float, default=0.1
        The size of the validation set used for early stopping
    cross_validate : bool, default=False
        Indicates whether to use cross-validation strategy for train/validation split
    early_stopping_patience : int or None, default=40
        The early stopping patience when early stopping is used.
        If set to None, no early stopping is employed
    num_model : int, default=1
        The total number of models used for Bagging strategy
    random_state : int or None, default=0
        Pseudo-random number generator to control the train/validation data split
        if early stoppingis enabled, the weight initialization, and the dropout.
        Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        Number of jobs to run in parallel. Training the estimator the score are parallelized
        over the number of models.
    device : {"cpu", "gpu"}, default="cpu",
        The device used for the estimator.
    disable_pbar : bool, default=True
        Indicates whether to show progress bars for the training process.
    """

    def __init__(
        self,
        *,
        loss: str = "squared_error",
        scoring: str = "r2_score",
        num_layers: int = 1,
        load_pretrain: bool = True,
        freeze_pretrain: bool = True,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        max_epoch: int = 500,
        dropout: float = 0,
        val_size: float = 0.2,
        cross_validate: bool = False,
        early_stopping_patience: Union[None, int] = 40,
        num_model: int = 1,
        random_state: int = 0,
        n_jobs: int = 1,
        device: str = "cpu",
        disable_pbar: bool = True,
        pretrained_model_path=config_directory["pretrained_model"],
    ):
        super(CARTERegressor, self).__init__(
            num_layers=num_layers,
            load_pretrain=load_pretrain,
            freeze_pretrain=freeze_pretrain,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epoch=max_epoch,
            dropout=dropout,
            val_size=val_size,
            cross_validate=cross_validate,
            early_stopping_patience=early_stopping_patience,
            num_model=num_model,
            random_state=random_state,
            n_jobs=n_jobs,
            device=device,
            disable_pbar=disable_pbar,
            pretrained_model_path=pretrained_model_path,
        )

        self.loss = loss
        self.scoring = scoring

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

        out = self._generate_output(X=X, model_list=self.model_list_, weights=None)

        return out


class CARTEClassifier(ClassifierMixin, BaseCARTEEstimator):
    """CARTE Classifier for Classification tasks.

    This estimator is GNN-based model compatible with the CARTE pretrained model.

    Parameters
    ----------
    loss : {'binary_crossentropy', 'categorical_crossentropy'}, default='binary_crossentropy'
        The loss function used for backpropagation.
    scoring : {'auroc', 'auprc', 'binary_entropy'}, default='auroc'
        The scoring function used for validation.
    num_layers : int, default=1
        The number of layers for the NN model
    load_pretrain : bool, default=True
        Indicates whether to load pretrained weights or not
    freeze_pretrain : bool, default=True
        Indicates whether to freeze the pretrained weights in the training or not
    learning_rate : float, default=1e-3
        The learning rate of the model. The model uses AdamW as the optimizer
    batch_size : int, default=16
        The batch size used for training
    max_epoch : int or None, default=500
        The maximum number of epoch for training
    dropout : float, default=0
        The dropout rate for training
    val_size : float, default=0.1
        The size of the validation set used for early stopping
    cross_validate : bool, default=False
        Indicates whether to use cross-validation strategy for train/validation split
    early_stopping_patience : int or None, default=40
        The early stopping patience when early stopping is used.
        If set to None, no early stopping is employed
    num_model : int, default=1
        The total number of models used for Bagging strategy
    random_state : int or None, default=0
        Pseudo-random number generator to control the train/validation data split
        if early stoppingis enabled, the weight initialization, and the dropout.
        Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        Number of jobs to run in parallel. Training the estimator the score are parallelized
        over the number of models.
    device : {"cpu", "gpu"}, default="cpu",
        The device used for the estimator.
    disable_pbar : bool, default=True
        Indicates whether to show progress bars for the training process.
    """

    def __init__(
        self,
        *,
        loss: str = "binary_crossentropy",
        scoring: str = "auroc",
        num_layers: int = 1,
        load_pretrain: bool = True,
        freeze_pretrain: bool = True,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        max_epoch: int = 500,
        dropout: float = 0,
        val_size: float = 0.2,
        cross_validate: bool = False,
        early_stopping_patience: Union[None, int] = 40,
        num_model: int = 1,
        random_state: int = 0,
        n_jobs: int = 1,
        device: str = "cpu",
        disable_pbar: bool = True,
        pretrained_model_path=config_directory["pretrained_model"],
    ):
        super(CARTEClassifier, self).__init__(
            num_layers=num_layers,
            load_pretrain=load_pretrain,
            freeze_pretrain=freeze_pretrain,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epoch=max_epoch,
            dropout=dropout,
            val_size=val_size,
            cross_validate=cross_validate,
            early_stopping_patience=early_stopping_patience,
            num_model=num_model,
            random_state=random_state,
            n_jobs=n_jobs,
            device=device,
            disable_pbar=disable_pbar,
            pretrained_model_path=pretrained_model_path,
        )

        self.loss = loss
        self.scoring = scoring

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

        if self.loss == "binary_crossentropy":
            return np.round(self.predict_proba(X))
        elif self.loss == "categorical_crossentropy":
            return np.argmax(self.predict_proba(X), axis=1)

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
        return self._get_predict_prob(X)

    def decision_function(self, X):
        """Compute the decision function of X.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
            The input samples.

        Returns
        -------
        decision : ndarray, shape (n_samples,)
        """
        decision = self.predict_proba(X)
        if decision.shape[1] == 1:
            decision = decision.ravel()
        return decision

    def _get_predict_prob(self, X):
        """Return the average of the outputs over all the models.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
            The input samples.

        Returns
        -------
        raw_predictions : array, shape (n_samples,)
            The raw predicted values.
        """

        out = self._generate_output(X=X, model_list=self.model_list_, weights=None)

        return out


class IdxIterator:
    """Class for iterating indices to set up the batch for CARTE Multitables"""

    def __init__(
        self,
        n_batch: int,
        domain_indicator: Tensor,
        target_fraction: float,
    ):
        self.n_batch = n_batch
        self.target_fraction = target_fraction
        self.domain_indicator = domain_indicator

        # Number of samples for target and source
        self.num_t = (domain_indicator == 0).sum().item()
        self.count_t = torch.ones(self.num_t)

        self.num_source_domain = domain_indicator.unique().size(0) - 1

        domain_list = domain_indicator.unique()
        source_domain_list = domain_list[domain_list != 0]

        self.num_s = [(domain_indicator == x).sum().item() for x in source_domain_list]

        count_s_ = [torch.ones(x) for x in self.num_s]
        self.count_s = count_s_[0]
        for x in range(1, self.num_source_domain):
            self.count_s = torch.block_diag(self.count_s, count_s_[x])
        if self.num_source_domain == 1:
            self.count_s = self.count_s.reshape(1, -1)
        self.count_s_fixed = copy.deepcopy(self.count_s)

        self.train_flag = None

        self.set_num_samples()

    def set_num_samples(self):
        self.num_samples_t = math.ceil(self.n_batch * self.target_fraction)
        n_batch_source_total = int((self.n_batch - self.num_samples_t))
        num_samples_s = [
            int(n_batch_source_total / self.num_source_domain)
            for _ in range(self.num_source_domain)
        ]
        if sum(num_samples_s) != n_batch_source_total:
            num_samples_s[
                torch.randint(0, self.num_source_domain, (1,))
            ] += n_batch_source_total - sum(num_samples_s)
        self.num_samples_s = num_samples_s

    def sample(self):
        idx_batch_t = torch.multinomial(
            self.count_t, num_samples=self.num_samples_t, replacement=False
        )
        self.count_t[idx_batch_t] -= 1

        idx_batch_s = torch.tensor([]).to(dtype=torch.long)
        for x in range(self.num_source_domain):
            idx_batch_s_ = torch.multinomial(
                self.count_s[x], num_samples=self.num_samples_s[x], replacement=False
            )
            self.count_s[x, idx_batch_s_] -= 1
            idx_batch_s = torch.hstack([idx_batch_s, idx_batch_s_])
            if torch.sum(self.count_s[x, :]) < self.num_samples_s[x]:
                self.count_s[x] = self.count_s_fixed[x, :]

        if torch.sum(self.count_t) < self.num_samples_t:
            self.count_t = torch.ones(self.num_t)
            self.train_flag = False

        return idx_batch_t, idx_batch_s


class BaseCARTEMultitableEstimator(BaseCARTEEstimator):
    """Base class for CARTE Multitable Estimator."""

    def __init__(
        self,
        *,
        source_data,
        num_layers,
        load_pretrain,
        freeze_pretrain,
        learning_rate,
        batch_size,
        max_epoch,
        dropout,
        val_size,
        target_fraction,
        early_stopping_patience,
        num_model,
        random_state,
        n_jobs,
        device,
        disable_pbar,
        pretrained_model_path,
    ):

        super(BaseCARTEMultitableEstimator, self).__init__(
            num_layers=num_layers,
            load_pretrain=load_pretrain,
            freeze_pretrain=freeze_pretrain,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epoch=max_epoch,
            dropout=dropout,
            val_size=val_size,
            early_stopping_patience=early_stopping_patience,
            num_model=num_model,
            random_state=random_state,
            n_jobs=n_jobs,
            device=device,
            disable_pbar=disable_pbar,
            pretrained_model_path=pretrained_model_path,
            cross_validate=False,  # overridden
        )

        self.source_data = source_data
        self.target_fraction = target_fraction

    def fit(self, X, y):
        """Fit the CARTE Multitable model.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
            The input samples of the target data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        # Preliminary settings
        self.is_fitted_ = False
        self.device_ = torch.device(self.device)
        self.X_ = X
        self.y_ = y
        self._set_task_specific_settings()

        # Set random_state, source list, and grid for parallelism
        random_state = check_random_state(self.random_state)
        random_state_list = [random_state.randint(1000) for _ in range(self.num_model)]
        self.source_list_total_ = list(self.source_data.keys()) + ["target"]
        grid = {"source": self.source_list_total_, "random_state": random_state_list}
        model_space_total = list(ParameterGrid(grid))

        # Fit model
        result_fit = Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_train_with_early_stopping)(model_space)
            for model_space in model_space_total
        )

        self.result_fit_ = result_fit

        # Store the required results that may be used later
        self.model_list_ = [model for (model, _, _, _) in result_fit]
        self.valid_loss_ = [valid_loss for (_, valid_loss, _, _) in result_fit]
        self.source_list_ = [sl for (_, _, sl, _) in result_fit]
        self.random_state_list_ = [rs for (_, _, _, rs) in result_fit]
        self.is_fitted_ = True

        val_loss_mean_ = []
        val_loss_total_ = []
        for source_name in self.source_list_total_:
            idx_ = np.where(np.array(self.source_list_) == source_name)[0]
            val_loss_total_ += [self.valid_loss_[idx] for idx in idx_]
            val_loss_mean_ += [np.array(val_loss_total_).mean()]
        val_loss_mean_ = -1 * np.array(val_loss_mean_)
        val_loss_total_ = -1 * np.array(val_loss_total_)
        weights = val_loss_mean_ / val_loss_total_.std()
        self.weights_ = np.exp(weights) / sum(np.exp(weights))

        return self

    def _run_train_with_early_stopping(self, model_space):
        """Train each model corresponding to the random_state with the early_stopping patience.

        This mode of training sets train/valid set for the early stopping criterion.
        Returns the trained model, train and validation loss at the best epoch, and the random_state.
        """

        # Set random_state and source data
        random_state = model_space["random_state"]
        if model_space["source"] == "target":
            target_only_flag = True
            source_data = None
        else:
            source_data = self.source_data[model_space["source"]]
            target_only_flag = False

        # Target dataset
        y_target = [data.y.cpu().detach().numpy() for data in self.X_]
        stratify = None
        if self._estimator_type == "classifier":
            stratify = y_target
        ds_train_target, ds_valid_target = train_test_split(
            self.X_,
            test_size=self.val_size,
            shuffle=True,
            stratify=stratify,
            random_state=random_state,
        )

        # Source dataset
        ds_train_source, ds_valid_source = self._set_source_data(
            source_data,
            ds_valid_target,
            random_state,
        )

        # Set validation batch for evaluation
        ds_valid = ds_valid_target + ds_valid_source
        ds_train = ds_train_target + ds_train_source
        ds_valid_eval = self._set_data_eval(data=ds_valid)

        # Load model and optimizer
        model_run_train = self._load_model()
        model_run_train.to(self.device_)
        optimizer = torch.optim.AdamW(
            model_run_train.parameters(), lr=self.learning_rate
        )

        # Train model
        valid_loss_best = 9e15
        es_counter = 0
        model_best_ = copy.deepcopy(model_run_train)

        if target_only_flag:
            train_loader = DataLoader(
                ds_train, batch_size=self.batch_size, shuffle=False
            )
        else:
            domain_indicator = torch.tensor([data.domain for data in ds_train])
            idx_iterator = IdxIterator(
                n_batch=self.batch_size,
                domain_indicator=domain_indicator,
                target_fraction=self.target_fraction,
            )

        for _ in tqdm(
            range(1, self.max_epoch + 1),
            desc=f"Model No. xx",
            disable=self.disable_pbar,
        ):

            # Run epoch
            if target_only_flag:
                self._run_epoch(model_run_train, optimizer, train_loader)
            else:
                self._run_epoch_multitable(
                    ds_train_source,
                    ds_train_target,
                    model_run_train,
                    optimizer,
                    idx_iterator,
                )

            # Obtain validation losses
            valid_loss = self._eval(model_run_train, ds_valid_eval)

            # Update model
            if valid_loss < valid_loss_best:
                valid_loss_best = valid_loss
                model_best_ = copy.deepcopy(model_run_train)
                es_counter = 0
            else:
                es_counter += 1
                if es_counter > self.early_stopping_patience:
                    break
        model_best_.eval()
        return model_best_, valid_loss_best, model_space["source"], random_state

    def _run_epoch_multitable(
        self,
        ds_source,
        ds_target,
        model,
        optimizer,
        idx_iterator,
    ):
        """Run an epoch for multitable of the input model."""
        model.train()
        idx_iterator.train_flag = True
        while idx_iterator.train_flag:
            idx_batch_target, idx_batch_source = idx_iterator.sample()
            ds_source_batch = [ds_source[idx] for idx in idx_batch_source]
            ds_target_batch = [ds_target[idx] for idx in idx_batch_target]
            ds_batch = ds_source_batch + ds_target_batch
            ds_train = self._set_data_eval(data=ds_batch)
            self._run_step(data=ds_train, model=model, optimizer=optimizer)

    def _set_source_data(self, source_data, ds_valid_target, random_state):
        """Prepare the source data for training."""
        if source_data is None:
            return [], []
        else:
            y_source = [data.y.cpu().detach().numpy() for data in source_data]
            stratify = [data.domain for data in source_data]
            stratify = np.array(stratify)
            if self._estimator_type == "classifier":
                y_source = [data.y.cpu().detach().numpy() for data in source_data]
                y_source = pd.Series(y_source)
                y_source = y_source.astype(str)
                stratify = pd.Series(stratify)
                stratify = stratify.astype(str)
                stratify = stratify + "_" + y_source
            ds_train_source, ds_valid_source = train_test_split(
                source_data,
                test_size=len(ds_valid_target),
                shuffle=True,
                stratify=stratify,
                random_state=random_state,
            )
            return ds_train_source, ds_valid_source


class CARTEMultitableRegressor(RegressorMixin, BaseCARTEMultitableEstimator):
    """CARTE Multitable Regressor for Regression tasks.

    This estimator is GNN-based model compatible with the CARTE pretrained model.

    Parameters
    ----------
    loss : {'squared_error', 'absolute_error'}, default='squared_error'
        The loss function used for backpropagation.
    scoring : {'r2_score', 'squared_error'}, default='r2_score'
        The scoring function used for validation.
    source_date : dict, default={}
        The source data used in multitable estimator.
    num_layers : int, default=1
        The number of layers for the NN model
    load_pretrain : bool, default=True
        Indicates whether to load pretrained weights or not
    freeze_pretrain : bool, default=True
        Indicates whether to freeze the pretrained weights in the training or not
    learning_rate : float, default=1e-3
        The learning rate of the model. The model uses AdamW as the optimizer
    batch_size : int, default=16
        The batch size used for training
    max_epoch : int or None, default=500
        The maximum number of epoch for training
    dropout : float, default=0
        The dropout rate for training
    val_size : float, default=0.1
        The size of the validation set used for early stopping
    target_fraction : float, default=0.125
        The fraction of target data inside of a batch when training
    early_stopping_patience : int or None, default=40
        The early stopping patience when early stopping is used.
        If set to None, no early stopping is employed
    num_model : int, default=1
        The total number of models used for Bagging strategy
    random_state : int or None, default=0
        Pseudo-random number generator to control the train/validation data split
        if early stoppingis enabled, the weight initialization, and the dropout.
        Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        Number of jobs to run in parallel. Training the estimator the score are parallelized
        over the number of models.
    device : {"cpu", "gpu"}, default="cpu",
        The device used for the estimator.
    disable_pbar : bool, default=True
        Indicates whether to show progress bars for the training process.
    """

    def __init__(
        self,
        *,
        loss: str = "squared_error",
        scoring: str = "r2_score",
        source_data: dict = {},
        num_layers: int = 1,
        load_pretrain: bool = True,
        freeze_pretrain: bool = True,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        max_epoch: int = 500,
        dropout: float = 0,
        val_size: float = 0.2,
        target_fraction: float = 0.125,
        early_stopping_patience: Union[None, int] = 40,
        num_model: int = 1,
        random_state: int = 0,
        n_jobs: int = 1,
        device: str = "cpu",
        disable_pbar: bool = True,
        pretrained_model_path=config_directory["pretrained_model"],
    ):
        super(CARTEMultitableRegressor, self).__init__(
            num_layers=num_layers,
            load_pretrain=load_pretrain,
            freeze_pretrain=freeze_pretrain,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epoch=max_epoch,
            dropout=dropout,
            val_size=val_size,
            early_stopping_patience=early_stopping_patience,
            num_model=num_model,
            random_state=random_state,
            n_jobs=n_jobs,
            device=device,
            disable_pbar=disable_pbar,
            pretrained_model_path=pretrained_model_path,
            source_data=source_data,
            target_fraction=target_fraction,
        )

        self.loss = loss
        self.scoring = scoring

    def predict(self, X):
        """Predict values for X.

        Returns the weighted average of the singletable model and all pairwise model with 1-source.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted values.
        """

        out = []
        for source_name in self.source_list_total_:
            idx_ = np.where(np.array(self.source_list_) == source_name)[0]
            model_list = [self.model_list_[idx] for idx in idx_]
            out += [self._generate_output(X, model_list=model_list, weights=None)]
        out = np.array(out).squeeze().transpose()
        out = np.average(out, weights=self.weights_, axis=1)
        if np.isnan(out).sum() > 0:
            mean_pred = np.mean(self.y_)
            out[np.isnan(out)] = mean_pred
        return out


class CARTEMultitableClassifer(ClassifierMixin, BaseCARTEMultitableEstimator):
    """CARTE Multitable Classifier for Classification tasks.

    This estimator is GNN-based model compatible with the CARTE pretrained model.

    Parameters
    ----------
    loss : {'binary_crossentropy', 'categorical_crossentropy'}, default='binary_crossentropy'
        The loss function used for backpropagation.
    scoring : {'auroc', 'auprc', 'binary_entropy'}, default='auroc'
        The scoring function used for validation.
    source_date : dict, default={}
        The source data used in multitable estimator.
    num_layers : int, default=1
        The number of layers for the NN model
    load_pretrain : bool, default=True
        Indicates whether to load pretrained weights or not
    freeze_pretrain : bool, default=True
        Indicates whether to freeze the pretrained weights in the training or not
    learning_rate : float, default=1e-3
        The learning rate of the model. The model uses AdamW as the optimizer
    batch_size : int, default=16
        The batch size used for training
    max_epoch : int or None, default=500
        The maximum number of epoch for training
    dropout : float, default=0
        The dropout rate for training
    val_size : float, default=0.1
        The size of the validation set used for early stopping
    target_fraction : float, default=0.125
        The fraction of target data inside of a batch when training
    early_stopping_patience : int or None, default=40
        The early stopping patience when early stopping is used.
        If set to None, no early stopping is employed
    num_model : int, default=1
        The total number of models used for Bagging strategy
    random_state : int or None, default=0
        Pseudo-random number generator to control the train/validation data split
        if early stoppingis enabled, the weight initialization, and the dropout.
        Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        Number of jobs to run in parallel. Training the estimator the score are parallelized
        over the number of models.
    device : {"cpu", "gpu"}, default="cpu",
        The device used for the estimator.
    disable_pbar : bool, default=True
        Indicates whether to show progress bars for the training process.
    """

    def __init__(
        self,
        *,
        loss: str = "binary_crossentropy",
        scoring: str = "auroc",
        source_data: dict = {},
        num_layers: int = 1,
        load_pretrain: bool = True,
        freeze_pretrain: bool = True,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        max_epoch: int = 500,
        dropout: float = 0,
        val_size: float = 0.2,
        target_fraction: float = 0.125,
        early_stopping_patience: Union[None, int] = 40,
        num_model: int = 1,
        random_state: int = 0,
        n_jobs: int = 1,
        device: str = "cpu",
        disable_pbar: bool = True,
        pretrained_model_path=config_directory["pretrained_model"],
    ):
        super(CARTEMultitableClassifer, self).__init__(
            num_layers=num_layers,
            load_pretrain=load_pretrain,
            freeze_pretrain=freeze_pretrain,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epoch=max_epoch,
            dropout=dropout,
            val_size=val_size,
            early_stopping_patience=early_stopping_patience,
            num_model=num_model,
            random_state=random_state,
            n_jobs=n_jobs,
            device=device,
            disable_pbar=disable_pbar,
            pretrained_model_path=pretrained_model_path,
            source_data=source_data,
            target_fraction=target_fraction,
        )

        self.loss = loss
        self.scoring = scoring

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

        if self.loss == "binary_crossentropy":
            return np.round(self.predict_proba(X))
        elif self.loss == "categorical_crossentropy":
            return np.argmax(self.predict_proba(X), axis=1)

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
        return self._get_predict_prob(X)

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
            The input samples.

        Returns
        -------
        decision : ndarray, shape (n_samples,)
        """
        decision = self.predict_proba(X)
        if decision.shape[1] == 1:
            decision = decision.ravel()
        return decision

    def _get_predict_prob(self, X):
        """Returns the weighted average of the singletable model and all pairwise model with 1-source.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
            The input samples.

        Returns
        -------
        raw_predictions : array, shape (n_samples,)
            The raw predicted values.
        """

        out = []
        for source_name in self.source_list_total_:
            idx_ = np.where(np.array(self.source_list_) == source_name)[0]
            model_list = [self.model_list_[idx] for idx in idx_]
            out += [self._generate_output(X, model_list=model_list, weights=None)]
        out = np.array(out).squeeze().transpose()
        out = np.average(out, weights=self.weights_, axis=1)
        # Transform according to loss
        if self.loss == "binary_crossentropy":
            out = 1 / (1 + np.exp(-out))
        elif self.loss == "categorical_crossentropy":
            out = softmax(out, axis=1)
        # Control for nulls in prediction
        if np.isnan(out).sum() > 0:
            mean_pred = np.mean(self.y_)
            out[np.isnan(out)] = mean_pred
        return out


class CARTE_AblationRegressor(CARTERegressor):
    """CARTE Ablation Regressor for Regression tasks.

    This estimator is GNN-based model compatible with the CARTE pretrained model.
    Note that this is an implementation for the ablation study of CARTE

    Parameters
    ----------
    ablation_method : {'exclude-edge', 'exclude-attention', 'exclude-attention-edge'}, default='exclude-edge'
        The ablation method for CARTE Estimators.
    loss : {'squared_error', 'absolute_error'}, default='squared_error'
        The loss function used for backpropagation.
    scoring : {'r2_score', 'squared_error'}, default='r2_score'
        The scoring function used for validation.
    num_layers : int, default=1
        The number of layers for the NN model
    load_pretrain : bool, default=True
        Indicates whether to load pretrained weights or not
    freeze_pretrain : bool, default=True
        Indicates whether to freeze the pretrained weights in the training or not
    learning_rate : float, default=1e-3
        The learning rate of the model. The model uses AdamW as the optimizer
    batch_size : int, default=16
        The batch size used for training
    max_epoch : int or None, default=500
        The maximum number of epoch for training
    dropout : float, default=0
        The dropout rate for training
    val_size : float, default=0.1
        The size of the validation set used for early stopping
    cross_validate : bool, default=False
        Indicates whether to use cross-validation strategy for train/validation split
    early_stopping_patience : int or None, default=40
        The early stopping patience when early stopping is used.
        If set to None, no early stopping is employed
    num_model : int, default=1
        The total number of models used for Bagging strategy
    random_state : int or None, default=0
        Pseudo-random number generator to control the train/validation data split
        if early stoppingis enabled, the weight initialization, and the dropout.
        Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        Number of jobs to run in parallel. Training the estimator the score are parallelized
        over the number of models.
    device : {"cpu", "gpu"}, default="cpu",
        The device used for the estimator.
    disable_pbar : bool, default=True
        Indicates whether to show progress bars for the training process.
    """

    def __init__(
        self,
        *,
        ablation_method: str = "exclude-edge",
        loss: str = "squared_error",
        scoring: str = "r2_score",
        num_layers: int = 1,
        load_pretrain: bool = True,
        freeze_pretrain: bool = True,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        max_epoch: int = 500,
        dropout: float = 0,
        val_size: float = 0.2,
        cross_validate: bool = False,
        early_stopping_patience: Union[None, int] = 40,
        num_model: int = 1,
        random_state: int = 0,
        n_jobs: int = 1,
        device: str = "cpu",
        disable_pbar: bool = True,
        pretrained_model_path=config_directory["pretrained_model"],
    ):

        super(CARTE_AblationRegressor, self).__init__(
            loss=loss,
            scoring=scoring,
            num_layers=num_layers,
            load_pretrain=load_pretrain,
            freeze_pretrain=freeze_pretrain,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epoch=max_epoch,
            dropout=dropout,
            val_size=val_size,
            cross_validate=cross_validate,
            early_stopping_patience=early_stopping_patience,
            num_model=num_model,
            random_state=random_state,
            n_jobs=n_jobs,
            device=device,
            disable_pbar=disable_pbar,
            pretrained_model_path=pretrained_model_path,
        )

        self.ablation_method = ablation_method

    def _load_model(self):
        """Load the CARTE Ablation model for training.

        This loads the pretrained weights if the parameter load_pretrain is set to True.
        The freeze of the pretrained weights are controlled by the freeze_pretrain parameter.

        Returns the model depending on the ablation method that can be used for training.
        """

        # Model configuration
        model_config = dict()
        model_config["ablation_method"] = self.ablation_method
        model_config["input_dim_x"] = self.X_[0].x.size(1)
        model_config["input_dim_e"] = self.X_[0].x.size(1)
        model_config["hidden_dim"] = self.X_[0].x.size(1)
        model_config["ff_dim"] = self.X_[0].x.size(1)
        model_config["num_heads"] = 12
        model_config["num_layers"] = self.num_layers - 1
        model_config["output_dim"] = self.output_dim_
        model_config["dropout"] = self.dropout

        # Set seed for torch - for reproducibility
        random_state = check_random_state(self.random_state)
        model_seed = random_state.randint(10000)
        torch.manual_seed(model_seed)

        # Set model architecture
        model = CARTE_NN_Model_Ablation(**model_config)

        # Load the pretrained weights if specified
        if self.load_pretrain and self.pretrained_model_path:
            # Use the specified pretrained model path
            model.load_state_dict(
                torch.load(self.pretrained_model_path, map_location=self.device_),
                strict=False,
            )
        # Freeze the pretrained weights if specified
        if self.freeze_pretrain:
            for param in model.ft_base.read_out_block.parameters():
                param.requires_grad = False
            for param in model.ft_base.layers.parameters():
                param.requires_grad = False

        # Set architecture for ablation
        if self.ablation_method == "exclude-edge":
            model.ft_base.initial_e = torch.nn.Identity()

        return model


class CARTE_AblationClassifier(CARTEClassifier):
    """CARTE Ablation Classifier for Classification tasks.

    This estimator is GNN-based model compatible with the CARTE pretrained model.
    Note that this is an implementation for the ablation study of CARTE

    Parameters
    ----------
    ablation_method : {'exclude-edge', 'exclude-attention', 'exclude-attention-edge'}, default='exclude-edge'
        The ablation method for CARTE Estimators.
    loss : {'binary_crossentropy', 'categorical_crossentropy'}, default='binary_crossentropy'
        The loss function used for backpropagation.
    scoring : {'auroc', 'auprc', 'binary_entropy'}, default='auroc'
        The scoring function used for validation.
    num_layers : int, default=1
        The number of layers for the NN model
    load_pretrain : bool, default=True
        Indicates whether to load pretrained weights or not
    freeze_pretrain : bool, default=True
        Indicates whether to freeze the pretrained weights in the training or not
    learning_rate : float, default=1e-3
        The learning rate of the model. The model uses AdamW as the optimizer
    batch_size : int, default=16
        The batch size used for training
    max_epoch : int or None, default=500
        The maximum number of epoch for training
    dropout : float, default=0
        The dropout rate for training
    val_size : float, default=0.1
        The size of the validation set used for early stopping
    cross_validate : bool, default=False
        Indicates whether to use cross-validation strategy for train/validation split
    early_stopping_patience : int or None, default=40
        The early stopping patience when early stopping is used.
        If set to None, no early stopping is employed
    num_model : int, default=1
        The total number of models used for Bagging strategy
    random_state : int or None, default=0
        Pseudo-random number generator to control the train/validation data split
        if early stoppingis enabled, the weight initialization, and the dropout.
        Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        Number of jobs to run in parallel. Training the estimator the score are parallelized
        over the number of models.
    device : {"cpu", "gpu"}, default="cpu",
        The device used for the estimator.
    disable_pbar : bool, default=True
        Indicates whether to show progress bars for the training process.
    """

    def __init__(
        self,
        *,
        ablation_method: str = "exclude-edge",
        loss: str = "binary_crossentropy",
        scoring: str = "auroc",
        num_layers: int = 1,
        load_pretrain: bool = False,
        freeze_pretrain: bool = False,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        max_epoch: int = 500,
        dropout: float = 0,
        val_size: float = 0.2,
        cross_validate: bool = False,
        early_stopping_patience: Union[None, int] = 40,
        num_model: int = 1,
        random_state: int = 0,
        n_jobs: int = 1,
        device: str = "cpu",
        disable_pbar: bool = True,
        pretrained_model_path=config_directory["pretrained_model"],
    ):

        super(CARTE_AblationClassifier, self).__init__(
            loss=loss,
            scoring=scoring,
            num_layers=num_layers,
            load_pretrain=load_pretrain,
            freeze_pretrain=freeze_pretrain,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epoch=max_epoch,
            dropout=dropout,
            val_size=val_size,
            cross_validate=cross_validate,
            early_stopping_patience=early_stopping_patience,
            num_model=num_model,
            random_state=random_state,
            n_jobs=n_jobs,
            device=device,
            disable_pbar=disable_pbar,
            pretrained_model_path=pretrained_model_path,
        )

        self.ablation_method = ablation_method

    def _load_model(self):
        """Load the CARTE Ablation model for training.

        This loads the pretrained weights if the parameter load_pretrain is set to True.
        The freeze of the pretrained weights are controlled by the freeze_pretrain parameter.

        Returns the model depending on the ablation method that can be used for training.
        """

        # Model configuration
        model_config = dict()
        model_config["ablation_method"] = self.ablation_method
        model_config["input_dim_x"] = self.X_[0].x.size(1)
        model_config["input_dim_e"] = self.X_[0].x.size(1)
        model_config["hidden_dim"] = self.X_[0].x.size(1)
        model_config["ff_dim"] = self.X_[0].x.size(1)
        model_config["num_heads"] = 12
        model_config["num_layers"] = self.num_layers - 1
        model_config["output_dim"] = self.output_dim_
        model_config["dropout"] = self.dropout

        # Set seed for torch - for reproducibility
        random_state = check_random_state(self.random_state)
        model_seed = random_state.randint(10000)
        torch.manual_seed(model_seed)

        # Set model architecture
        model = CARTE_NN_Model_Ablation(**model_config)

        # Load the pretrained weights if specified
        if self.load_pretrain and self.pretrained_model_path:
            # Load the pretrained model from the specified path
            pretrain_model_dict = torch.load(
                self.pretrained_model_path, map_location=self.device_
            )

            # Rename the keys containing "initial_x" to "initial_x_pretrain"
            initial_x_keys = [
                key for key in pretrain_model_dict.keys() if "initial_x" in key
            ]
            for key in initial_x_keys:
                pretrain_model_dict[key + "_pretrain"] = pretrain_model_dict.pop(key)

            # Load the state dict into the model
            model.load_state_dict(pretrain_model_dict, strict=False)

        # Freeze the pretrained weights if specified
        if self.freeze_pretrain:
            for param in model.ft_base.read_out_block.parameters():
                param.requires_grad = False
            for param in model.ft_base.layers.parameters():
                param.requires_grad = False

        # Set architecture for ablation
        if self.ablation_method == "exclude-edge":
            model.ft_base.initial_e = torch.nn.Identity()

        return model