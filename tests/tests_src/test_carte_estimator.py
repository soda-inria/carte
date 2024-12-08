import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from carte_ai import (
    CARTERegressor,
    CARTEClassifier,
    CARTE_AblationRegressor,
    CARTE_AblationClassifier,
    CARTEMultitableRegressor,
    CARTEMultitableClassifer,
)
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
    y = np.random.rand(100) * 100  # Regression target
    X = pd.DataFrame(X)
    y = pd.Series(y)
    # Ensure column names are strings
    X.columns = X.columns.astype(str)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def dummy_spotify_data():
    """Create dummy Spotify dataset for classification."""
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = X.columns.astype(str)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def table_to_graph_transformer(fasttext_model_path):
    """Load the Table2GraphTransformer with a pre-trained FastText model."""
    return Table2GraphTransformer(fasttext_model_path=fasttext_model_path)


@pytest.mark.parametrize("num_model", [1, 5, 10])
@pytest.mark.parametrize("device", ["cpu"])
def test_carte_regressor_fit_predict(dummy_wine_data, num_model, device, table_to_graph_transformer):
    X_train, X_test, y_train, y_test = dummy_wine_data
    X_train = X_train.astype(str)
    X_test = X_test.astype(str)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

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
    assert y_pred.shape == y_test.shape


@pytest.mark.parametrize("num_model", [1, 5, 10])
@pytest.mark.parametrize("device", ["cpu"])
def test_carte_classifier_fit_predict(dummy_spotify_data, num_model, device, table_to_graph_transformer):
    X_train, X_test, y_train, y_test = dummy_spotify_data
    X_train = X_train.astype(str)
    X_test = X_test.astype(str)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

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
    assert y_pred_proba.shape[0] == y_test.shape[0]


@pytest.mark.parametrize("cross_validate", [True, False])
def test_carte_cross_validation(dummy_wine_data, cross_validate, table_to_graph_transformer):
    X_train, X_test, y_train, y_test = dummy_wine_data
    X_train = X_train.astype(str)
    X_test = X_test.astype(str)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

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
    assert y_pred.shape == y_test.shape


@pytest.fixture
def dummy_multiclass_data():
    np.random.seed(42)
    X = np.random.rand(60, 5)
    y = np.random.randint(0, 3, 60)
    X = pd.DataFrame(X)
    X.columns = X.columns.astype(str)
    X = X.astype(str)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def dummy_source_data(dummy_wine_data):
    return {"source_domain_1": []}


def add_domain_attribute(graph_list, domain_value):
    for g in graph_list:
        g.domain = domain_value
    return graph_list


def test_ablation_regressor(dummy_wine_data, table_to_graph_transformer):
    X_train, X_test, y_train, y_test = dummy_wine_data
    X_train = X_train.astype(str)
    X_test = X_test.astype(str)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    X_train_transformed = table_to_graph_transformer.fit_transform(X_train, y=y_train)
    X_test_transformed = table_to_graph_transformer.transform(X_test)

    regressor = CARTE_AblationRegressor(
        ablation_method="exclude-edge",
        loss="absolute_error",
        scoring="squared_error",
        random_state=42,
        disable_pbar=True,
        pretrained_model_path=config_directory["pretrained_model"]
    )
    regressor.fit(X_train_transformed, y_train)
    y_pred = regressor.predict(X_test_transformed)
    assert y_pred.shape == y_test.shape


def test_ablation_classifier_multiclass(dummy_multiclass_data, table_to_graph_transformer):
    X_train, X_test, y_train, y_test = dummy_multiclass_data
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    X_train_transformed = table_to_graph_transformer.fit_transform(X_train, y=y_train)
    X_test_transformed = table_to_graph_transformer.transform(X_test)

    classifier = CARTE_AblationClassifier(
        ablation_method="exclude-edge",
        loss="categorical_crossentropy",
        scoring="auroc",
        random_state=42,
        disable_pbar=True,
        pretrained_model_path=config_directory["pretrained_model"]
    )
    classifier.fit(X_train_transformed, y_train)
    y_pred_proba = classifier.predict_proba(X_test_transformed)
    assert y_pred_proba.shape == (len(y_test), 3)
    np.testing.assert_almost_equal(y_pred_proba.sum(axis=1), np.ones(len(y_test)), decimal=2)


# def test_multitable_regressor(dummy_wine_data, dummy_source_data, table_to_graph_transformer):
#     X_train, X_test, y_train, y_test = dummy_wine_data
#     X_train = X_train.astype(str)
#     X_test = X_test.astype(str)
#     y_train = y_train.to_numpy()
#     y_test = y_test.to_numpy()

#     X_train_transformed = table_to_graph_transformer.fit_transform(X_train, y=y_train)
#     X_train_transformed = add_domain_attribute(X_train_transformed, domain_value=0)
#     X_test_transformed = table_to_graph_transformer.transform(X_test)
#     X_test_transformed = add_domain_attribute(X_test_transformed, domain_value=0)

#     source_data_key = list(dummy_source_data.keys())[0]
#     source_samples = X_train.sample(30, random_state=42)
#     source_samples = source_samples.astype(str)
#     source_transformed = table_to_graph_transformer.transform(source_samples)
#     source_transformed = add_domain_attribute(source_transformed, domain_value=1)
#     dummy_source_data[source_data_key] = source_transformed

#     # Ensure scaler is passed to multitable training by updating the source code:
#     # In carte_estimator.py, inside `_run_train_with_early_stopping` for multitable:
#     #     scaler = amp.GradScaler()
#     #     ...
#     # And in `_run_epoch_multitable` method, add `scaler` as argument and pass it to `_run_step()`.

#     regressor = CARTEMultitableRegressor(
#         source_data=dummy_source_data,
#         random_state=42,
#         disable_pbar=True,
#         pretrained_model_path=config_directory["pretrained_model"]
#     )
#     regressor.fit(X_train_transformed, y_train)
#     y_pred = regressor.predict(X_test_transformed)
#     assert y_pred.shape == y_test.shape


# def test_multitable_classifier(dummy_spotify_data, dummy_source_data, table_to_graph_transformer):
#     X_train, X_test, y_train, y_test = dummy_spotify_data
#     X_train = X_train.astype(str)
#     X_test = X_test.astype(str)
#     y_train = y_train.to_numpy()
#     y_test = y_test.to_numpy()

#     X_train_transformed = table_to_graph_transformer.fit_transform(X_train, y=y_train)
#     X_train_transformed = add_domain_attribute(X_train_transformed, domain_value=0)
#     X_test_transformed = table_to_graph_transformer.transform(X_test)
#     X_test_transformed = add_domain_attribute(X_test_transformed, domain_value=0)

#     source_data_key = list(dummy_source_data.keys())[0]
#     source_samples = X_train.sample(30, random_state=42)
#     source_samples = source_samples.astype(str)
#     source_transformed = table_to_graph_transformer.transform(source_samples)
#     source_transformed = add_domain_attribute(source_transformed, domain_value=1)
#     dummy_source_data[source_data_key] = source_transformed

#     # Similarly ensure `scaler` is passed in `_run_train_with_early_stopping` and `_run_epoch_multitable`.

#     classifier = CARTEMultitableClassifer(
#         source_data=dummy_source_data,
#         random_state=42,
#         disable_pbar=True,
#         pretrained_model_path=config_directory["pretrained_model"]
#     )
#     classifier.fit(X_train_transformed, y_train)
#     y_pred_proba = classifier.predict_proba(X_test_transformed)
#     assert y_pred_proba.shape[0] == y_test.shape[0]
