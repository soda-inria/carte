import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from carte_ai import CARTERegressor, CARTEClassifier
from carte_ai.src.carte_table_to_graph import Table2GraphTransformer
from carte_ai.configs.directory import config_directory
from huggingface_hub import hf_hub_download


@pytest.fixture(scope="module")
def fasttext_model_path():
    """Download FastText model from Hugging Face Hub."""
    return hf_hub_download(repo_id="hi-paris/fastText", filename="cc.en.300.bin")


@pytest.fixture
def dummy_wine_data():
    """Create dummy Wine Poland dataset."""
    np.random.seed(42)
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    y = np.random.rand(100) * 100  # Regression target (price prediction)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    # Ensure column names are strings to avoid issues with .str accessor
    X.columns = X.columns.astype(str)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def dummy_spotify_data():
    """Create dummy Spotify dataset for classification."""
    np.random.seed(42)
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    y = np.random.randint(0, 2, 100)  # Binary classification target (popularity)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    # Ensure column names are strings to avoid issues with .str accessor
    X.columns = X.columns.astype(str)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def table_to_graph_transformer(fasttext_model_path):
    """Load the Table2GraphTransformer with a pre-trained FastText model."""
    return Table2GraphTransformer(fasttext_model_path=fasttext_model_path)


@pytest.mark.parametrize("num_model", [1, 5, 10])
@pytest.mark.parametrize("device", ["cpu"])
def test_carte_regressor_fit_predict(dummy_wine_data, num_model, device, table_to_graph_transformer):
    """Test fitting and predicting with CARTERegressor on Wine dataset."""
    X_train, X_test, y_train, y_test = dummy_wine_data

    # Convert columns to string objects
    X_train = X_train.astype(str)
    X_test = X_test.astype(str)

    # Convert y to numpy arrays
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # Transform the table data into graphs
    X_train_transformed = table_to_graph_transformer.fit_transform(X_train, y=y_train)
    X_test_transformed = table_to_graph_transformer.transform(X_test)

    fixed_params = {
        "num_model": num_model,
        "disable_pbar": True,
        "random_state": 42,
        "device": device,
        "n_jobs": 1,
        "pretrained_model_path": config_directory["pretrained_model"],
    }

    regressor = CARTERegressor(**fixed_params)
    regressor.fit(X_train_transformed, y_train)
    y_pred = regressor.predict(X_test_transformed)

    # Just check shape and that predictions run through
    assert y_pred.shape == y_test.shape, f"Expected shape {y_test.shape}, but got {y_pred.shape}"
    # Removed the R2 >= 0 assertion, as random data may produce negative R².


@pytest.mark.parametrize("num_model", [1, 5, 10])
@pytest.mark.parametrize("device", ["cpu"])
def test_carte_classifier_fit_predict(dummy_spotify_data, num_model, device, table_to_graph_transformer):
    """Test fitting and predicting with CARTEClassifier on Spotify dataset."""
    X_train, X_test, y_train, y_test = dummy_spotify_data

    # Convert columns to string objects
    X_train = X_train.astype(str)
    X_test = X_test.astype(str)

    # Convert y to numpy arrays
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # Transform the table data into graphs
    X_train_transformed = table_to_graph_transformer.fit_transform(X_train, y=y_train)
    X_test_transformed = table_to_graph_transformer.transform(X_test)

    fixed_params = {
        "num_model": num_model,
        "disable_pbar": True,
        "random_state": 42,
        "device": device,
        "n_jobs": 1,
        "pretrained_model_path": config_directory["pretrained_model"],
    }

    classifier = CARTEClassifier(**fixed_params)
    classifier.fit(X_train_transformed, y_train)
    y_pred_proba = classifier.predict_proba(X_test_transformed)

    assert y_pred_proba.shape[0] == y_test.shape[0], f"Expected {y_test.shape[0]} predictions, but got {y_pred_proba.shape[0]}"

    # Check the dimensionality of y_pred_proba before indexing
    if y_pred_proba.ndim == 1:
        # If we have a single dimension, treat them as probabilities for the positive class
        score = roc_auc_score(y_test, y_pred_proba)
    else:
        # If two-dimensional, proceed as originally
        score = roc_auc_score(y_test, y_pred_proba[:, 1])

    # Removed assertion that AUROC >= 0, since AUROC is always between 0 and 1 anyway.


@pytest.mark.parametrize("cross_validate", [True, False])
def test_carte_cross_validation(dummy_wine_data, cross_validate, table_to_graph_transformer):
    """Test CARTERegressor with and without cross-validation on Wine dataset."""
    X_train, X_test, y_train, y_test = dummy_wine_data

    # Convert columns to string objects
    X_train = X_train.astype(str)
    X_test = X_test.astype(str)

    # Convert y to numpy arrays
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # Transform the table data into graphs
    X_train_transformed = table_to_graph_transformer.fit_transform(X_train, y=y_train)
    X_test_transformed = table_to_graph_transformer.transform(X_test)

    fixed_params = {
        "num_model": 5,
        "disable_pbar": True,
        "random_state": 42,
        "device": "cpu",
        "n_jobs": 1,
        "pretrained_model_path": config_directory["pretrained_model"],
        "cross_validate": cross_validate,
        "val_size": 0.2,
    }

    regressor = CARTERegressor(**fixed_params)
    regressor.fit(X_train_transformed, y_train)
    y_pred = regressor.predict(X_test_transformed)

    # Just check shape and that predictions run through
    assert y_pred.shape == y_test.shape, f"Expected shape {y_test.shape}, but got {y_pred.shape}"
    # Removed the R2 >= 0 assertion here as well, due to random data.
