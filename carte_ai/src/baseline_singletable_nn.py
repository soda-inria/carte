"""Neural network baseline for comparison."""

import torch
import torch.nn as nn
import numpy as np
import copy
from typing import Union
from torch import Tensor
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted, check_random_state
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from joblib import Parallel, delayed


## Simple MLP model
class MLP_Model(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_prob: float,
        num_layers: int,
    ):
        super().__init__()

        self.initial = nn.Linear(input_dim, hidden_dim)

        self.mlp_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        self.layers = nn.Sequential(*[self.mlp_block for _ in range(num_layers)])

        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        X = self.initial(X)
        X = self.layers(X)
        X = self.classifier(X)
        return X


## Residual Block
class Residual_Block(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_factor: int,
        normalization: Union[str, None] = "layernorm",
        hidden_dropout_prob: float = 0.2,
        residual_dropout_prob: float = 0.2,
    ):
        super().__init__()

        self.lin1 = nn.Linear(input_dim, output_dim * hidden_factor)
        self.lin2 = nn.Linear(output_dim * hidden_factor, output_dim)
        self.relu = nn.ReLU()
        self.dropout_hidden = nn.Dropout(hidden_dropout_prob)
        self.dropout_residual = nn.Dropout(residual_dropout_prob)

        self.norm1: Union[nn.BatchNorm1d, nn.LayerNorm, None]
        self.norm2: Union[nn.BatchNorm1d, nn.LayerNorm, None]
        if normalization == "batchnorm":
            self.norm1 = nn.BatchNorm1d(output_dim * hidden_factor)
            self.norm2 = nn.BatchNorm1d(output_dim)
        elif normalization == "layernorm":
            self.norm1 = nn.LayerNorm(output_dim * hidden_factor)
            self.norm2 = nn.LayerNorm(output_dim)
        else:
            self.norm1 = self.norm2 = None

    def reset_parameters(self) -> None:
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()

    def forward(self, x: Tensor):
        out = self.lin1(x)
        out = self.norm1(out) if self.norm1 else out
        out = self.relu(out)
        out = self.dropout_hidden(out)

        out = self.lin2(out)
        out = self.norm2(out) if self.norm2 else out
        out = self.relu(out)
        out = self.dropout_residual(out)

        out = out + x
        out = self.relu(out)

        return out


## Resnet model
class RESNET_Model(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        **block_args,
    ):
        super(RESNET_Model, self).__init__()

        self.initial = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [
                Residual_Block(
                    input_dim=hidden_dim, output_dim=hidden_dim, **block_args
                )
                for _ in range(num_layers)
            ]
        )

        self.classifer = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        X = self.initial(X)

        for l in self.layers:
            X = l(X)

        X = self.classifer(X)
        return X


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPBase(BaseEstimator):
    """Base class for MLP."""

    def __init__(
        self,
        *,
        hidden_dim,
        learning_rate,
        weight_decay,
        batch_size,
        val_size,
        num_model,
        max_epoch,
        early_stopping_patience,
        n_jobs,
        device,
        random_state,
        disable_pbar,
    ):
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.val_size = val_size
        self.num_model = num_model
        self.max_epoch = max_epoch
        self.early_stopping_patience = early_stopping_patience
        self.n_jobs = n_jobs
        self.device = device
        self.random_state = random_state
        self.disable_pbar = disable_pbar

    def fit(self, X, y):
        # Preliminary settings
        self.is_fitted_ = False
        self.device_ = torch.device(self.device)
        self.X_ = X
        self.y_ = y
        self._set_task_specific_settings()

        if isinstance(X, Tensor) == False:
            X = torch.tensor(X, dtype=torch.float32)
        if isinstance(y, Tensor) == False:
            y = torch.tensor(y, dtype=torch.float32)

        # Set random_state
        random_state = check_random_state(self.random_state)
        random_state_list = [random_state.randint(1000) for _ in range(self.num_model)]

        # Fit model
        result_fit = Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_train_with_early_stopping)(X, y, rs)
            for rs in random_state_list
        )

        # Store the required results that may be used later
        self.model_list_ = [model for (model, _, _) in result_fit]
        self.valid_loss_ = [valid_loss for (_, valid_loss, _) in result_fit]
        self.random_state_list_ = [rs for (_, _, rs) in result_fit]
        self.is_fitted_ = True

        return self

    def _run_train_with_early_stopping(self, X, y, random_state):
        """Train each model corresponding to the random_state with the early_stopping patience.

        This mode of training sets train/valid set for the early stopping criterion.
        Returns the trained model, train and validation loss at the best epoch, and the random_state.
        """
        # Set validation by val_size
        stratify = None
        if self.model_task_ == "classification":
            stratify = self.y_
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=self.val_size,
            shuffle=True,
            random_state=random_state,
            stratify=stratify,
        )

        ds_train = TabularDataset(X_train, y_train)

        # Load model and optimizer
        input_dim = X.size(1)
        model_run_train = self._load_model(input_dim)
        model_run_train.to(self.device_)
        optimizer = torch.optim.AdamW(
            model_run_train.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Train model
        train_loader = DataLoader(ds_train, batch_size=self.batch_size, shuffle=False)
        valid_loss_best = 9e15

        es_counter = 0
        model_best_ = copy.deepcopy(model_run_train)
        for _ in tqdm(
            range(1, self.max_epoch + 1),
            desc=f"Model No. {random_state}",
            disable=self.disable_pbar,
        ):
            self._run_epoch(model_run_train, optimizer, train_loader)
            valid_loss = self._eval(model_run_train, X_valid, y_valid)
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

    def _run_epoch(self, model, optimizer, train_loader):
        """Run an epoch of the input model.

        With each epoch, it updates the model and the optimizer.
        """
        model.train()
        for data_X, data_y in train_loader:
            optimizer.zero_grad()  # Clear gradients.
            data_X = data_X.to(self.device_)
            data_y = data_y.to(self.device_)
            out = model(data_X)  # Perform a single forward pass.
            target = data_y
            out = out.view(-1).to(torch.float64)
            target = target.to(torch.float64)
            loss = self.criterion_(out, target)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.

    def _eval(self, model, X, y):
        """Run an evaluation of the input data on the input model.

        Returns the selected loss of the input data from the input model.
        """
        X = X.to(self.device_)
        y = y.to(self.device_)
        with torch.no_grad():
            model.eval()
            out = model(X)
            target = y
            out = out.view(-1).to(torch.float64)
            target = target.to(torch.float64)
            loss_eval = self.criterion_(out, target)
            loss_eval = round(loss_eval.detach().item(), 4)
        return loss_eval

    def _set_task_specific_settings(self):
        self.criterion_ = None
        self.output_dim_ = None
        self.model_task_ = None

    def _load_model(self, input_dim):
        return None


class BaseMLPEstimator(MLPBase):
    """Base class for MLP Estimator."""

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout_prob: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        batch_size: int = 128,
        val_size: float = 0.1,
        num_model: int = 1,
        max_epoch: int = 200,
        early_stopping_patience: Union[None, int] = 10,
        n_jobs: int = 1,
        device: str = "cpu",
        random_state: int = 0,
        disable_pbar: bool = True,
    ):
        super(BaseMLPEstimator, self).__init__(
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            val_size=val_size,
            num_model=num_model,
            max_epoch=max_epoch,
            early_stopping_patience=early_stopping_patience,
            n_jobs=n_jobs,
            device=device,
            random_state=random_state,
            disable_pbar=disable_pbar,
        )

        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

    def _load_model(self, input_dim):
        """Load the MLP model for training.

        Returns the model that can be used for training.
        """

        # Set seed for torch - for reproducibility
        random_state = check_random_state(self.random_state)
        model_seed = random_state.randint(10000)
        torch.manual_seed(model_seed)

        model_config = dict()
        model_config["input_dim"] = input_dim
        model_config["hidden_dim"] = self.hidden_dim
        model_config["output_dim"] = self.output_dim_
        model_config["dropout_prob"] = self.dropout_prob
        model_config["num_layers"] = self.num_layers
        model = MLP_Model(**model_config)
        return model


class MLPRegressor(RegressorMixin, BaseMLPEstimator):
    """ """

    def __init__(
        self,
        *,
        loss: str = "squared_error",
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout_prob: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        batch_size: int = 128,
        val_size: float = 0.1,
        num_model: int = 1,
        max_epoch: int = 200,
        early_stopping_patience: Union[None, int] = 10,
        n_jobs: int = 1,
        device: str = "cpu",
        random_state: int = 0,
        disable_pbar: bool = True,
    ):
        super(MLPRegressor, self).__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            val_size=val_size,
            num_model=num_model,
            max_epoch=max_epoch,
            early_stopping_patience=early_stopping_patience,
            n_jobs=n_jobs,
            device=device,
            random_state=random_state,
            disable_pbar=disable_pbar,
        )

        self.loss = loss

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        if isinstance(X, Tensor) == False:
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device_)

        # Obtain the predicitve output
        with torch.no_grad():
            out = [model(X).cpu().detach().numpy() for model in self.model_list_]

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
        if self.loss == "squared_error":
            self.criterion_ = torch.nn.MSELoss()
        elif self.loss == "absolute_error":
            self.criterion_ = torch.nn.L1Loss()

        self.output_dim_ = 1
        self.model_task_ = "regression"


class MLPClassifier(ClassifierMixin, BaseMLPEstimator):
    """ """

    def __init__(
        self,
        *,
        loss: str = "binary_crossentropy",
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout_prob: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        batch_size: int = 128,
        val_size: float = 0.1,
        num_model: int = 1,
        max_epoch: int = 200,
        early_stopping_patience: Union[None, int] = 10,
        n_jobs: int = 1,
        device: str = "cpu",
        random_state: int = 0,
        disable_pbar: bool = True,
    ):
        super(MLPClassifier, self).__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            val_size=val_size,
            num_model=num_model,
            max_epoch=max_epoch,
            early_stopping_patience=early_stopping_patience,
            n_jobs=n_jobs,
            device=device,
            random_state=random_state,
            disable_pbar=disable_pbar,
        )

        self.loss = loss

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        if isinstance(X, Tensor) == False:
            X = torch.tensor(X, dtype=torch.float32)

        if self.loss == "binary_crossentropy":
            return np.round(self.predict_proba(X))
        elif self.loss == "categorical_crossentropy":
            return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        if isinstance(X, Tensor) == False:
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device_)
        return self._get_predict_prob(X)

    def decision_function(self, X):
        decision = self.predict_proba(X)
        if decision.shape[1] == 1:
            decision = decision.ravel()
        return decision

    def _get_predict_prob(self, X):
        # Obtain the predicitve output
        with torch.no_grad():
            out = [model(X).cpu().detach().numpy() for model in self.model_list_]
        out = np.mean(out, axis=0)
        if self.loss == "binary_crossentropy":
            out = 1 / (1 + np.exp(-out))
        elif self.loss == "categorical_crossentropy":
            out = np.exp(out) / sum(np.exp(out))
        return out

    def _set_task_specific_settings(self):
        if self.loss == "binary_crossentropy":
            self.criterion_ = torch.nn.BCEWithLogitsLoss()
        elif self.loss == "categorical_crossentropy":
            self.criterion_ = torch.nn.CrossEntropyLoss()

        self.output_dim_ = len(np.unique(self.y_))
        if self.output_dim_ == 2:
            self.output_dim_ -= 1
            self.criterion_ = torch.nn.BCEWithLogitsLoss()

        self.model_task_ = "classification"


class BaseRESNETEstimator(MLPBase):
    """Base class for RESNET Estimator."""

    def __init__(
        self,
        *,
        normalization: Union[str, None] = "layernorm",
        num_layers: int = 4,
        hidden_dim: int = 256,
        hidden_factor: int = 2,
        hidden_dropout_prob: float = 0.2,
        residual_dropout_prob: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        batch_size: int = 128,
        val_size: float = 0.1,
        num_model: int = 1,
        max_epoch: int = 200,
        early_stopping_patience: Union[None, int] = 10,
        n_jobs: int = 1,
        device: str = "cpu",
        random_state: int = 0,
        disable_pbar: bool = True,
    ):
        super(BaseRESNETEstimator, self).__init__(
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            val_size=val_size,
            num_model=num_model,
            max_epoch=max_epoch,
            early_stopping_patience=early_stopping_patience,
            n_jobs=n_jobs,
            device=device,
            random_state=random_state,
            disable_pbar=disable_pbar,
        )

        self.normalization = normalization
        self.num_layers = num_layers
        self.hidden_factor = hidden_factor
        self.hidden_dropout_prob = hidden_dropout_prob
        self.residual_dropout_prob = residual_dropout_prob

    def _load_model(self, input_dim):
        """Load the RESNET model for training.

        Returns the model that can be used for training.
        """

        # Set seed for torch - for reproducibility
        random_state = check_random_state(self.random_state)
        model_seed = random_state.randint(10000)
        torch.manual_seed(model_seed)

        model_config = dict()
        model_config["input_dim"] = input_dim
        model_config["hidden_dim"] = self.hidden_dim
        model_config["output_dim"] = self.output_dim_
        model_config["hidden_factor"] = self.hidden_factor
        model_config["hidden_dropout_prob"] = self.hidden_dropout_prob
        model_config["residual_dropout_prob"] = self.residual_dropout_prob
        model_config["normalization"] = self.normalization
        model_config["num_layers"] = self.num_layers

        model = RESNET_Model(**model_config)
        return model


class RESNETRegressor(RegressorMixin, BaseRESNETEstimator):
    """ """

    def __init__(
        self,
        *,
        loss: str = "squared_error",
        normalization: Union[str, None] = "layernorm",
        num_layers: int = 4,
        hidden_dim: int = 256,
        hidden_factor: int = 2,
        hidden_dropout_prob: float = 0.2,
        residual_dropout_prob: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        batch_size: int = 128,
        val_size: float = 0.1,
        num_model: int = 1,
        max_epoch: int = 200,
        early_stopping_patience: Union[None, int] = 10,
        n_jobs: int = 1,
        device: str = "cpu",
        random_state: int = 0,
        disable_pbar: bool = True,
    ):
        super(RESNETRegressor, self).__init__(
            normalization=normalization,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            hidden_factor=hidden_factor,
            hidden_dropout_prob=hidden_dropout_prob,
            residual_dropout_prob=residual_dropout_prob,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            val_size=val_size,
            num_model=num_model,
            max_epoch=max_epoch,
            early_stopping_patience=early_stopping_patience,
            n_jobs=n_jobs,
            device=device,
            random_state=random_state,
            disable_pbar=disable_pbar,
        )

        self.loss = loss

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        if isinstance(X, Tensor) == False:
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device_)

        # Obtain the predicitve output
        with torch.no_grad():
            out = [model(X).cpu().detach().numpy() for model in self.model_list_]

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
        if self.loss == "squared_error":
            self.criterion_ = torch.nn.MSELoss()
        elif self.loss == "absolute_error":
            self.criterion_ = torch.nn.L1Loss()

        self.output_dim_ = 1
        self.model_task_ = "regression"


class RESNETClassifier(ClassifierMixin, BaseRESNETEstimator):
    """ """

    def __init__(
        self,
        *,
        loss: str = "binary_crossentropy",
        normalization: Union[str, None] = "layernorm",
        num_layers: int = 4,
        hidden_dim: int = 256,
        hidden_factor: int = 2,
        hidden_dropout_prob: float = 0.2,
        residual_dropout_prob: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        batch_size: int = 128,
        val_size: float = 0.1,
        num_model: int = 1,
        max_epoch: int = 200,
        early_stopping_patience: Union[None, int] = 10,
        n_jobs: int = 1,
        device: str = "cpu",
        random_state: int = 0,
        disable_pbar: bool = True,
    ):
        super(RESNETClassifier, self).__init__(
            normalization=normalization,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            hidden_factor=hidden_factor,
            hidden_dropout_prob=hidden_dropout_prob,
            residual_dropout_prob=residual_dropout_prob,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            val_size=val_size,
            num_model=num_model,
            max_epoch=max_epoch,
            early_stopping_patience=early_stopping_patience,
            n_jobs=n_jobs,
            device=device,
            random_state=random_state,
            disable_pbar=disable_pbar,
        )

        self.loss = loss

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        if isinstance(X, Tensor) == False:
            X = torch.tensor(X, dtype=torch.float32)

        if self.loss == "binary_crossentropy":
            return np.round(self.predict_proba(X))
        elif self.loss == "categorical_crossentropy":
            return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        if isinstance(X, Tensor) == False:
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device_)
        return self._get_predict_prob(X)

    def decision_function(self, X):
        decision = self.predict_proba(X)
        if decision.shape[1] == 1:
            decision = decision.ravel()
        return decision

    def _get_predict_prob(self, X):
        # Obtain the predicitve output
        with torch.no_grad():
            out = [model(X).cpu().detach().numpy() for model in self.model_list_]
        out = np.mean(out, axis=0)
        if self.loss == "binary_crossentropy":
            out = 1 / (1 + np.exp(-out))
        elif self.loss == "categorical_crossentropy":
            out = np.exp(out) / sum(np.exp(out))
        return out

    def _set_task_specific_settings(self):
        if self.loss == "binary_crossentropy":
            self.criterion_ = torch.nn.BCEWithLogitsLoss()
        elif self.loss == "categorical_crossentropy":
            self.criterion_ = torch.nn.CrossEntropyLoss()

        self.output_dim_ = len(np.unique(self.y_))
        if self.output_dim_ == 2:
            self.output_dim_ -= 1
            self.criterion_ = torch.nn.BCEWithLogitsLoss()

        self.model_task_ = "classification"
