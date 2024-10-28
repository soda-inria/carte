import pytest
import torch
import numpy as np
from sklearn.metrics import r2_score, roc_auc_score
from carte_ai import CARTERegressor, CARTEClassifier, Table2GraphTransformer
from sklearn.model_selection import train_test_split
from carte_ai.configs.directory import config_directory

@pytest.fixture
def dummy_wine_data():
    """Create dummy Wine Poland dataset."""
    np.random.seed(42)
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    y = np.random.rand(100) * 100  # Regression target (price prediction)
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def dummy_spotify_data():
    """Create dummy Spotify dataset for classification."""
    np.random.seed(42)
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    y = np.random.randint(0, 2, 100)  # Binary classification target (popularity)
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def table_to_graph_transformer():
    """Load a pre-trained FastText model to simulate the Table2Graph transformation."""
    return Table2GraphTransformer(fasttext_model_path="path/to/fasttext.bin")

@pytest.mark.parametrize("num_model", [1, 5, 10])
@pytest.mark.parametrize("device", ["cpu"])
def test_carte_regressor_fit_predict(dummy_wine_data, num_model, device, table_to_graph_transformer):
    """Test fitting and predicting with CARTERegressor on Wine dataset."""
    X_train, X_test, y_train, y_test = dummy_wine_data

    # Transform the table data into graphs using Table2GraphTransformer
    X_train_transformed = table_to_graph_transformer.fit_transform(X_train, y=y_train)
    X_test_transformed = table_to_graph_transformer.transform(X_test)

    # Set fixed parameters for the estimator
    fixed_params = {
        "num_model": num_model,
        "disable_pbar": True,
        "random_state": 42,
        "device": device,
        "n_jobs": 1,
        "pretrained_model_path": config_directory["pretrained_model"]
    }

    # Initialize the regressor
    regressor = CARTERegressor(**fixed_params)

    # Fit the model
    regressor.fit(X_train_transformed, y_train)

    # Predict
    y_pred = regressor.predict(X_test_transformed)

    # Check predictions shape and compute R2 score
    assert y_pred.shape == y_test.shape, f"Expected shape {y_test.shape}, but got {y_pred.shape}"
    score = r2_score(y_test, y_pred)
    assert score >= 0, "R2 score should be non-negative"
    print(f"R2 score for num_model={num_model}, device={device}: {score:.4f}")

@pytest.mark.parametrize("num_model", [1, 5, 10])
@pytest.mark.parametrize("device", ["cpu"])
def test_carte_classifier_fit_predict(dummy_spotify_data, num_model, device, table_to_graph_transformer):
    """Test fitting and predicting with CARTEClassifier on Spotify dataset."""
    X_train, X_test, y_train, y_test = dummy_spotify_data

    # Transform the table data into graphs using Table2GraphTransformer
    X_train_transformed = table_to_graph_transformer.fit_transform(X_train, y=y_train)
    X_test_transformed = table_to_graph_transformer.transform(X_test)

    # Set fixed parameters for the estimator
    fixed_params = {
        "num_model": num_model,
        "disable_pbar": True,
        "random_state": 42,
        "device": device,
        "n_jobs": 1,
        "pretrained_model_path": config_directory["pretrained_model"]
    }

    # Initialize the classifier
    classifier = CARTEClassifier(**fixed_params)

    # Fit the model
    classifier.fit(X_train_transformed, y_train)

    # Predict probabilities
    y_pred_proba = classifier.predict_proba(X_test_transformed)

    # Check predictions shape and compute AUROC score
    assert y_pred_proba.shape[0] == y_test.shape[0], f"Expected {y_test.shape[0]} predictions, but got {y_pred_proba.shape[0]}"
    score = roc_auc_score(y_test, y_pred_proba[:, 1])  # Use the second column for AUROC
    assert score >= 0, "AUROC score should be non-negative"
    print(f"AUROC score for num_model={num_model}, device={device}: {score:.4f}")

@pytest.mark.parametrize("cross_validate", [True, False])
def test_carte_cross_validation(dummy_wine_data, cross_validate, table_to_graph_transformer):
    """Test CARTERegressor with and without cross-validation on Wine dataset."""
    X_train, X_test, y_train, y_test = dummy_wine_data

    # Transform the table data into graphs using Table2GraphTransformer
    X_train_transformed = table_to_graph_transformer.fit_transform(X_train, y=y_train)
    X_test_transformed = table_to_graph_transformer.transform(X_test)

    # Set parameters for the estimator
    fixed_params = {
        "num_model": 5,
        "disable_pbar": True,
        "random_state": 42,
        "device": "cpu",
        "n_jobs": 1,
        "pretrained_model_path": config_directory["pretrained_model"],
        "cross_validate": cross_validate,
        "val_size": 0.2
    }

    # Initialize the regressor
    regressor = CARTERegressor(**fixed_params)

    # Fit the model
    regressor.fit(X_train_transformed, y_train)

    # Predict
    y_pred = regressor.predict(X_test_transformed)

    # Check predictions shape and compute R2 score
    assert y_pred.shape == y_test.shape, f"Expected shape {y_test.shape}, but got {y_pred.shape}"
    score = r2_score(y_test, y_pred)
    assert score >= 0, "R2 score should be non-negative"
    print(f"Cross-validation={cross_validate}, R2 score: {score:.4f}")
