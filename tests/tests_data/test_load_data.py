import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from carte_ai.data.load_data import (
    load_parquet_config,
    set_split,
    set_split_hf,
    spotify,
    wina_pl,
    wine_dot_com_prices,
    wine_vivino_price,
)

# Mocked DataFrame
mock_data = pd.DataFrame({
    "entity_name": ["A", "B", "C", "D"],
    "target_name": [1, 0, 1, 0],
    "feature1": [10, 20, 30, 40],
    "feature2": [100, 200, 300, 400]
})

# Mocked Configurations
mock_config = {
    "entity_name": "entity_name",
    "target_name": "target_name",
    "task": "regression",
    "repeated": False
}

### Test `load_parquet_config` ###
@patch("carte_ai.data.load_data.requests.get")
@patch("pandas.read_parquet")
def test_load_parquet_config(mock_read_parquet, mock_requests):
    # Mock parquet loading
    mock_read_parquet.return_value = mock_data

    # Mock config JSON
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_config
    mock_requests.return_value = mock_response

    data, config = load_parquet_config("mock_dataset")

    # Assertions
    mock_read_parquet.assert_called_once()
    mock_requests.assert_called_once()
    assert isinstance(data, pd.DataFrame), "Data should be a DataFrame"
    assert isinstance(config, dict), "Config should be a dictionary"
    assert config == mock_config, "Config does not match expected values"

### Test `set_split` ###
def test_set_split():
    X_train, X_test, y_train, y_test = set_split(mock_data, mock_config, num_train=2)

    # Assertions
    assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
    assert isinstance(X_test, pd.DataFrame), "X_test should be a DataFrame"
    assert isinstance(y_train, np.ndarray), "y_train should be a NumPy array"
    assert isinstance(y_test, np.ndarray), "y_test should be a NumPy array"
    assert len(X_train) > 0 and len(X_test) > 0, "Train and test splits should not be empty"

### Test `set_split_hf` ###
def test_set_split_hf():
    X_train, X_test, y_train, y_test = set_split_hf(
        mock_data, target_name="target_name", entity_name="entity_name", num_train=2
    )

    # Assertions
    assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
    assert isinstance(X_test, pd.DataFrame), "X_test should be a DataFrame"
    assert isinstance(y_train, np.ndarray), "y_train should be a NumPy array"
    assert isinstance(y_test, np.ndarray), "y_test should be a NumPy array"
    assert len(X_train) > 0 and len(X_test) > 0, "Train and test splits should not be empty"

### Test dataset-specific methods ###
@patch("carte_ai.data.load_data.load_parquet_config", return_value=(mock_data, mock_config))
def test_spotify(mock_load_parquet_config):
    X_train, X_test, y_train, y_test = spotify(num_train=2)
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0, "Spotify train/test splits should not be empty"

@patch("carte_ai.data.load_data.load_parquet_config", return_value=(mock_data, mock_config))
def test_wina_pl(mock_load_parquet_config):
    X_train, X_test, y_train, y_test = wina_pl(num_train=2)
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0, "Wina_PL train/test splits should not be empty"

@patch("carte_ai.data.load_data.load_parquet_config", return_value=(mock_data, mock_config))
def test_wine_dot_com_prices(mock_load_parquet_config):
    X_train, X_test, y_train, y_test = wine_dot_com_prices(num_train=2)
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0, "Wine.com Prices train/test splits should not be empty"

@patch("carte_ai.data.load_data.load_parquet_config", return_value=(mock_data, mock_config))
def test_wine_vivino_price(mock_load_parquet_config):
    X_train, X_test, y_train, y_test = wine_vivino_price(num_train=2)
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0, "Vivino Wine Prices train/test splits should not be empty"
