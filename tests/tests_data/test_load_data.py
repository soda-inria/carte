import pytest
import pandas as pd
import json
from carte_ai.data.load_data import *

# Parametrize with additional test cases including edge cases
@pytest.mark.parametrize("data_path, config_path", [
    ("/mnt/data/wine_vivino_price.parquet", "/mnt/data/config_wine_vivino_price.json"),
    ("/mnt/data/wine_pl.parquet", "/mnt/data/config_wine_pl.json"),
    ("/mnt/data/wine_dot_com_prices.parquet", "/mnt/data/config_wine_dot_com_prices.json"),
    ("/mnt/data/spotify.parquet", "/mnt/data/config_spotify.json"),
    # Edge case: Non-existent data file
    ("invalid/path/non_existent.parquet", "/mnt/data/config_wine_vivino_price.json"),
    # Edge case: Missing config file
    ("/mnt/data/wine_vivino_price.parquet", "invalid/path/non_existent_config.json"),
])
def test_load_data(data_path, config_path):
    try:
        # Load the configuration
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        # Test the behavior when config file is missing
        pytest.fail(f"Configuration file not found: {config_path}")
        return

    # Try loading the data
    try:
        data = load_data(config)
    except FileNotFoundError:
        # Test the behavior when data file is missing
        pytest.fail(f"Data file not found: {data_path}")
        return
    except Exception as e:
        pytest.fail(f"Exception occurred while loading data: {str(e)}")
        return

    # Basic assertions to ensure the data is loaded correctly
    assert isinstance(data, pd.DataFrame), "The loaded data is not a DataFrame."
    assert config["entity_name"] in data.columns, f"{config['entity_name']} not found in DataFrame columns."
    assert config["target_name"] in data.columns, f"{config['target_name']} not found in DataFrame columns."
    assert not data.empty, "The DataFrame is empty."

    # Check for missing values
    assert data.isna().sum().sum() == 0, "There are missing values in the DataFrame."

    # Add task-specific checks
    if config["task"] == "regression":
        assert pd.api.types.is_numeric_dtype(data[config["target_name"]]), "Target for regression is not numeric."
    elif config["task"] == "classification":
        assert data[config["target_name"]].nunique() > 1, "Target for classification has only one class."

    # Check for duplicates if specified in config
    if not config.get("repeated", False):
        assert not data.duplicated(subset=[config["entity_name"]]).any(), "There are duplicate entries in the entity column."

@pytest.mark.parametrize("config,expected_exception", [
    # Malformed config: missing entity_name
    ({"target_name": "Price", "task": "regression"}, KeyError),
    # Malformed config: missing target_name
    ({"entity_name": "Name", "task": "regression"}, KeyError),
    # Malformed config: invalid task type
    ({"entity_name": "Name", "target_name": "Price", "task": "invalid_task"}, ValueError),
])
def test_load_data_invalid_configs(config, expected_exception):
    # Test behavior with malformed configuration
    with pytest.raises(expected_exception):
        load_data(config)
