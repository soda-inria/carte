"""CARTE multitable estimator for regression and classification."""

import torch
import math
import numpy as np
import pandas as pd
import copy
from typing import Union
from torch import Tensor
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_random_state
from joblib import Parallel, delayed
from tqdm import tqdm

from nn_model.carte_downstream import CARTE_NN_Model
from configs.directory import config_directory


# Index Iterator for batch
class IdxIterator:
    """Class for iterating indices to set up the batch"""

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


# Estimators
class BaseCARTEMultitableEstimator(BaseEstimator):
    """Base class for CARTE Estimator."""

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
    ):
        self.source_data = source_data
        self.num_layers = num_layers
        self.load_pretrain = load_pretrain
        self.freeze_pretrain = freeze_pretrain
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.dropout = dropout
        self.val_size = val_size
        self.target_fraction = target_fraction
        self.early_stopping_patience = early_stopping_patience
        self.num_model = num_model
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.device = device
        self.disable_pbar = disable_pbar

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

        # Set random_state
        random_state = check_random_state(self.random_state)
        random_state_list = [random_state.randint(1000) for _ in range(self.num_model)]

        # Fit model
        result_fit = Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_train_with_early_stopping)(rs) for rs in random_state_list
        )

        # Store the required results that may be used later
        self.model_list_ = [model for (model, _, _) in result_fit]
        self.valid_loss_ = [valid_loss for (_, valid_loss, _) in result_fit]
        self.random_state_list_ = [rs for (_, _, rs) in result_fit]
        self.is_fitted_ = True

        return self

    def _run_train_with_early_stopping(self, random_state):
        """Train each model corresponding to the random_state with the early_stopping patience.

        This mode of training sets train/valid set for the early stopping criterion.
        Returns the trained model, train and validation loss at the best epoch, and the random_state.
        """

        # Target dataset
        y_target = [data.y.cpu().detach().numpy() for data in self.X_]
        stratify = None
        if self.model_task_ == "classification":
            stratify = y_target
        ds_train_target, ds_valid_target = train_test_split(
            self.X_,
            test_size=self.val_size,
            shuffle=True,
            stratify=stratify,
            random_state=random_state,
        )

        # Source dataset
        y_source = [data.y.cpu().detach().numpy() for data in self.source_data]
        stratify = [data.domain for data in self.source_data]
        stratify = np.array(stratify)
        if self.model_task_ == "classification":
            y_source = [data.y.cpu().detach().numpy() for data in self.source_data]
            y_source = pd.Series(y_source)
            y_source = y_source.astype(str)
            stratify = pd.Series(stratify)
            stratify = stratify.astype(str)
            stratify = stratify + "_" + y_source
        ds_train_source, ds_valid_source = train_test_split(
            self.source_data,
            test_size=len(ds_valid_target),
            shuffle=True,
            stratify=stratify,
            random_state=random_state,
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
        domain_indicator = torch.tensor([data.domain for data in ds_train])

        idx_iterator = IdxIterator(
            n_batch=self.batch_size,
            domain_indicator=domain_indicator,
            target_fraction=self.target_fraction,
        )

        for _ in tqdm(
            range(1, self.max_epoch + 1),
            desc=f"Model No. {random_state}",
            disable=self.disable_pbar,
        ):
            # Run epoch
            self._run_epoch(
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
        return model_best_, valid_loss_best, random_state

    def _run_epoch(self, ds_source, ds_target, model, optimizer, idx_iterator):
        """Run an epoch of the input model."""
        model.train()
        idx_iterator.train_flag = True
        while idx_iterator.train_flag:
            idx_batch_target, idx_batch_source = idx_iterator.sample()
            ds_source_batch = [ds_source[idx] for idx in idx_batch_source]
            ds_target_batch = [ds_target[idx] for idx in idx_batch_target]
            ds_batch = ds_source_batch + ds_target_batch
            ds_train = self._set_data_eval(data=ds_batch)
            self._run_step(data=ds_train, model=model, optimizer=optimizer)

    def _run_step(self, data, model, optimizer):
        """Run a step of the input model.

        With each step, it updates the model and the optimizer.
        """
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        # data.to(self.device_)  # Send to device
        out = model(data)  # Perform a single forward pass.
        target = data.y  # Set target
        out = out.view(-1).to(torch.float32)  # Reshape outputSet head index
        target = target.to(torch.float32)  # Reshape target
        loss = self.criterion_(out, target)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

    def _eval(self, model, ds_eval):
        """Run an evaluation of the input data on the input model.

        Returns the selected loss of the input data from the input model.
        """
        with torch.no_grad():
            model.eval()
            out = model(ds_eval)
            target = ds_eval.y
            out = out.view(-1).to(torch.float32)
            target = target.to(torch.float32)
            loss_eval = self.criterion_(out, target)
            loss_eval = loss_eval.detach().item()
        return loss_eval

    def _set_data_eval(self, data):
        """Construct the aggregated graph object from the list of data.

        This is consistent with the graph object from torch_geometric.
        Returns the aggregated graph object.
        """
        make_batch = Batch()
        with torch.no_grad():
            ds_eval = make_batch.from_data_list(data, follow_batch=["edge_index"])
            ds_eval.to(self.device_)
        return ds_eval

    def _set_task_specific_settings(self):
        """Set task specific settings for regression and classfication.

        This is overridden in each of the subclasses.
        """
        self.criterion_ = None
        self.output_dim_ = None
        self.model_task_ = None

    def _load_model(self):
        """Load the CARTE model for training.

        This loads the pretrained weights if the parameter load_pretrain is set to True.
        The freeze of the pretrained weights are controlled by the freeze_pretrain parameter.

        Returns the model that can be used for training.
        """
        # Model configuration
        model_config = dict()
        model_config["input_dim_x"] = 300
        model_config["input_dim_e"] = 300
        model_config["hidden_dim"] = 300
        model_config["ff_dim"] = 300
        model_config["num_heads"] = 12
        model_config["num_layers"] = self.num_layers
        model_config["output_dim"] = self.output_dim_
        model_config["dropout"] = self.dropout

        # Set seed for torch - for reproducibility
        random_state = check_random_state(self.random_state)
        model_seed = random_state.randint(10000)
        torch.manual_seed(model_seed)

        # Set model architecture
        model = CARTE_NN_Model(**model_config)

        # Load the pretrained weights if specified
        if self.load_pretrain:
            dir_model = config_directory["pretrained_model"]
            model.load_state_dict(
                torch.load(dir_model, map_location=self.device_), strict=False
            )
        # Freeze the pretrained weights if specified
        if self.freeze_pretrain:
            for param in model.ft_base.read_out_block.parameters():
                param.requires_grad = False
            for param in model.ft_base.layers.parameters():
                param.requires_grad = False

        return model


class CARTEMultitableRegressor(RegressorMixin, BaseCARTEMultitableEstimator):
    """CARTE Regressor for Regression tasks.

    This estimator is GNN-based model compatible with the CARTE pretrained model.

    Parameters
    ----------
    loss : {'squared_error', 'absolute_error'}, default='squared_error'
        The loss function used for backpropagation.
    num_layers : int, default=0
        The number of layers for the NN model
    load_pretrain : bool, default=True
        Indicates whether to load pretrained weights or not
    freeze_pretrain : bool, default=True
        Indicates whether to freeze the pretrained weights in the training or not
    learning_rate : float, default=1e-3
        The learning rate of the model. The model uses AdamW as the optimizer
    batch_size : int, default=64
        The batch size used for training
    max_epoch : int or None, default=1000
        The maximum number of epoch for training
    dropout : float, default=0
        The dropout rate for training
    val_size : float, default=0.1
        The size of the validation set used for early stopping
    target_fraction : float, default=0.125
        The fraction of target data inside of a batch when training
    early_stopping_patience : int or None, default=100
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
        source_data: Union[None, list] = None,
        num_layers: int = 0,
        load_pretrain: bool = True,
        freeze_pretrain: bool = True,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        max_epoch: int = 1000,
        dropout: float = 0.2,
        val_size: float = 0.2,
        target_fraction: float = 0.125,
        early_stopping_patience: Union[None, int] = 100,
        num_model: int = 1,
        random_state: int = 0,
        n_jobs: int = 1,
        device: str = "cpu",
        disable_pbar: bool = True,
    ):
        super(CARTEMultitableRegressor, self).__init__(
            source_data=source_data,
            num_layers=num_layers,
            load_pretrain=load_pretrain,
            freeze_pretrain=freeze_pretrain,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epoch=max_epoch,
            dropout=dropout,
            val_size=val_size,
            target_fraction=target_fraction,
            early_stopping_patience=early_stopping_patience,
            num_model=num_model,
            random_state=random_state,
            n_jobs=n_jobs,
            device=device,
            disable_pbar=disable_pbar,
        )

        self.loss = loss

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

        # Obtain the batch to feed into the network
        ds_predict_eval = self._set_data_eval(data=X)

        # Obtain the predicitve output
        with torch.no_grad():
            out = [
                model(ds_predict_eval).cpu().detach().numpy()
                for model in self.model_list_
            ]

        if self.num_model == 1:
            out = np.array(out).squeeze().transpose()
        else:
            out = np.array(out).squeeze().transpose()
            out = np.mean(out, axis=1)

        return out

    def _set_task_specific_settings(self):
        """Set task specific settings for regression and classfication.

        Sets the loss, output dimension, and the task.
        """
        if self.loss == "squared_error":
            self.criterion_ = torch.nn.MSELoss()
        elif self.loss == "absolute_error":
            self.criterion_ = torch.nn.L1Loss()

        self.output_dim_ = 1
        self.model_task_ = "regression"


class CARTEMultitableClassifier(ClassifierMixin, BaseCARTEMultitableEstimator):
    """CARTE Classifier for Classification tasks.

    This estimator is GNN-based model compatible with the CARTE pretrained model.

    Parameters
    ----------
    loss : {'binary_crossentropy', 'categorical_crossentropy'}, default='binary_crossentropy'
        The loss function used for backpropagation.
    num_layers : int, default=0
        The number of layers for the NN model
    load_pretrain : bool, default=True
        Indicates whether to load pretrained weights or not
    freeze_pretrain : bool, default=False
        Indicates whether to freeze the pretrained weights in the training or not
    learning_rate : float, default=1e-3
        The learning rate of the model. The model uses AdamW as the optimizer
    batch_size : int, default=64
        The batch size used for training
    max_epoch : int or None, default=1000
        The maximum number of epoch for training
    dropout : float, default=0
        The dropout rate for training
    val_size : float, default=0.1
        The size of the validation set used for early stopping
    target_fraction : float, default=0.125
        The fraction of target data inside of a batch when training
    early_stopping_patience : int or None, default=100
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
        source_data: Union[None, list] = None,
        num_layers: int = 0,
        load_pretrain: bool = True,
        freeze_pretrain: bool = True,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        max_epoch: int = 1000,
        dropout: float = 0.2,
        val_size: float = 0.2,
        target_fraction: float = 0.125,
        early_stopping_patience: Union[None, int] = 100,
        num_model: int = 1,
        random_state: int = 0,
        n_jobs: int = 1,
        device: str = "cpu",
        disable_pbar: bool = True,
    ):
        super(CARTEMultitableClassifier, self).__init__(
            source_data=source_data,
            num_layers=num_layers,
            load_pretrain=load_pretrain,
            freeze_pretrain=freeze_pretrain,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epoch=max_epoch,
            dropout=dropout,
            val_size=val_size,
            target_fraction=target_fraction,
            early_stopping_patience=early_stopping_patience,
            num_model=num_model,
            random_state=random_state,
            n_jobs=n_jobs,
            device=device,
            disable_pbar=disable_pbar,
        )

        self.loss = loss

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

        # Obtain the batch to feed into the network
        ds_predict_eval = self._set_data_eval(data=X)

        # Obtain the predicitve output
        with torch.no_grad():
            out = [
                model(ds_predict_eval).cpu().detach().numpy()
                for model in self.model_list_
            ]
        out = np.mean(out, axis=0)
        if self.loss == "binary_crossentropy":
            out = 1 / (1 + np.exp(-out))
        elif self.loss == "categorical_crossentropy":
            out = np.exp(out) / sum(np.exp(out))
        return out

    def _set_task_specific_settings(self):
        """Set task specific settings for regression and classfication.

        Sets the loss, output dimension, and the task.
        """
        if self.loss == "binary_crossentropy":
            self.criterion_ = torch.nn.BCEWithLogitsLoss()
        elif self.loss == "categorical_crossentropy":
            self.criterion_ = torch.nn.CrossEntropyLoss()

        self.output_dim_ = len(np.unique(self.y_))
        if self.output_dim_ == 2:
            self.output_dim_ -= 1
            self.criterion_ = torch.nn.BCEWithLogitsLoss()

        self.classes_ = np.unique(self.y_)
        self.model_task_ = "classification"
