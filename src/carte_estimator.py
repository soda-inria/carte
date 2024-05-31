"""CARTE estimator for regression and classification."""

import torch
import numpy as np
import copy
from typing import Union
from torcheval.metrics import (
    MeanSquaredError,
    R2Score,
    BinaryAUROC,
    BinaryAUPRC,
    MulticlassAUROC,
)
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_random_state
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.special import softmax
from src.carte_model import CARTE_NN_Model
from configs.directory import config_directory


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
        early_stopping_patience,
        num_model,
        random_state,
        n_jobs,
        device,
        disable_pbar,
    ):
        self.num_layers = num_layers
        self.load_pretrain = load_pretrain
        self.freeze_pretrain = freeze_pretrain
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.dropout = dropout
        self.val_size = val_size
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

        # Set the cv-splits
        n_splits = int(1 / self.val_size)
        n_repeats = int(self.num_model / n_splits)

        if self.model_task_ == "regression":
            rfk = RepeatedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=self.random_state
            )
            splits = [
                (train_index, test_index)
                for train_index, test_index in rfk.split(np.arange(0, len(X)))
            ]
        else:
            rfk = RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=self.random_state
            )
            splits = [
                (train_index, test_index)
                for train_index, test_index in rfk.split(np.arange(0, len(X)), y)
            ]

        # Fit model
        result_fit = Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_train_with_early_stopping)(X, split_index)
            for split_index in splits
        )

        # Store the required results that may be used later
        self.model_list_ = [model for (model, _) in result_fit]
        self.valid_loss_ = [valid_loss for (_, valid_loss) in result_fit]
        self.is_fitted_ = True

        return self

    def _run_train_with_early_stopping(self, X, split_index):
        """Train each model corresponding to the random_state with the early_stopping patience.

        This mode of training sets train/valid set for the early stopping criterion.
        Returns the trained model, train and validation loss at the best epoch, and the random_state.
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
            self._run_epoch(model_run_train, optimizer, train_loader)
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

    def _run_epoch(self, model, optimizer, train_loader):
        """Run an epoch of the input model.

        With each epoch, it updates the model and the optimizer.
        """
        model.train()
        for data in train_loader:  # Iterate in batches over the training dataset.
            optimizer.zero_grad()  # Clear gradients.
            data.to(self.device_)  # Send to device
            out = model(data)  # Perform a single forward pass.
            target = data.y  # Set target
            if self.output_dim_ == 1:
                out = out.view(-1).to(torch.float32)  # Reshape outputSet head index
                target = target.to(torch.float32)  # Reshape target
            loss = self.criterion_(out, target)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            # optimizer.zero_grad()  # Clear gradients.

    def _eval(self, model, ds_eval):
        """Run an evaluation of the input data on the input model.

        Returns the selected loss of the input data from the input model.
        """
        with torch.no_grad():
            model.eval()
            out = model(ds_eval)
            target = ds_eval.y
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
        self.valid_loss_metric_ = None
        self.valid_loss_flag_ = None
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
        model_config["input_dim_x"] = self.X_[0].x.size(1)
        model_config["input_dim_e"] = self.X_[0].x.size(1)
        model_config["hidden_dim"] = self.X_[0].x.size(1)
        model_config["ff_dim"] = self.X_[0].x.size(1)
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
            pretrain_model_dict = torch.load(dir_model, map_location=self.device_)
            initial_x_keys = [
                key for key in pretrain_model_dict.keys() if "initial_x" in key
            ]
            for key in initial_x_keys:
                pretrain_model_dict[key + "_pretrain"] = pretrain_model_dict.pop(key)
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
    num_layers : int, default=0
        The number of layers for the NN model
    load_pretrain : bool, default=True
        Indicates whether to load pretrained weights or not
    freeze_pretrain : bool, default=False
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
        num_layers: int = 0,
        load_pretrain: bool = True,
        freeze_pretrain: bool = True,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        max_epoch: int = 500,
        dropout: float = 0,
        val_size: float = 0.2,
        early_stopping_patience: Union[None, int] = 40,
        num_model: int = 1,
        random_state: int = 0,
        n_jobs: int = 1,
        device: str = "cpu",
        disable_pbar: bool = True,
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
            early_stopping_patience=early_stopping_patience,
            num_model=num_model,
            random_state=random_state,
            n_jobs=n_jobs,
            device=device,
            disable_pbar=disable_pbar,
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

        if np.isnan(out).sum() > 0:
            mean_pred = np.mean(self.y_)
            out[np.isnan(out)] = mean_pred

        return out

    def _set_task_specific_settings(self):
        """Set task specific settings for regression and classfication.

        Sets the loss, output dimension, and the task.
        """
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

        self.valid_loss_metric_.to(self.device_)
        self.output_dim_ = 1
        self.model_task_ = "regression"


class CARTEClassifier(ClassifierMixin, BaseCARTEEstimator):
    """CARTE Classifier for Classification tasks.

    This estimator is GNN-based model compatible with the CARTE pretrained model.

    Parameters
    ----------
    loss : {'binary_crossentropy', 'categorical_crossentropy'}, default='binary_crossentropy'
        The loss function used for backpropagation.
    scoring : {'auroc', 'auprc'}, default='auroc'
        The scoring function used for validation.
    num_layers : int, default=0
        The number of layers for the NN model
    load_pretrain : bool, default=True
        Indicates whether to load pretrained weights or not
    freeze_pretrain : bool, default=False
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
        num_layers: int = 0,
        load_pretrain: bool = True,
        freeze_pretrain: bool = True,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        max_epoch: int = 500,
        dropout: float = 0,
        val_size: float = 0.2,
        early_stopping_patience: Union[None, int] = 40,
        num_model: int = 1,
        random_state: int = 0,
        n_jobs: int = 1,
        device: str = "cpu",
        disable_pbar: bool = True,
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
            early_stopping_patience=early_stopping_patience,
            num_model=num_model,
            random_state=random_state,
            n_jobs=n_jobs,
            device=device,
            disable_pbar=disable_pbar,
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
            out = softmax(out, axis=1)

        if np.isnan(out).sum() > 0:
            mean_pred = np.mean(self.y_)
            out[np.isnan(out)] = mean_pred

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

        if self.scoring == "auroc":
            self.valid_loss_metric_ = BinaryAUROC()
            self.valid_loss_flag_ = "neg"
        elif self.scoring == "auprc":
            self.valid_loss_metric_ = BinaryAUPRC()
            self.valid_loss_flag_ = "neg"

        if self.loss == "categorical_crossentropy":
            self.valid_loss_metric_ = MulticlassAUROC(num_classes=self.output_dim_)
            self.valid_loss_flag_ = "neg"

        self.classes_ = np.unique(self.y_)
        self.model_task_ = "classification"
