import pytest
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from huggingface_hub import hf_hub_download
from carte_ai.src.carte_table_to_graph import Table2GraphTransformer, _create_edge_index


@pytest.fixture(scope="module")
def fasttext_model_path():
    """Download FastText model from Hugging Face Hub."""
    return hf_hub_download(repo_id="hi-paris/fastText", filename="cc.en.300.bin")


@pytest.fixture
def dummy_data():
    """Create dummy dataset with both numerical and categorical columns."""
    num_data = {
        'feature1': np.random.rand(10),  # Numerical feature
        'feature2': np.random.rand(10),  # Numerical feature
        'category': ['A', 'B', 'A', 'B', 'C', 'C', 'A', 'B', 'C', 'A']  # Categorical feature
    }
    return pd.DataFrame(num_data)


@pytest.fixture
def dummy_labels():
    """Create dummy labels."""
    return np.random.randint(0, 2, 10)  # Binary labels


@pytest.fixture
def large_dummy_data():
    """Create a large dummy dataset."""
    num_data = {
        'feature1': np.random.rand(10000),  # Numerical feature
        'feature2': np.random.rand(10000),  # Numerical feature
        'category': np.random.choice(['A', 'B', 'C'], 10000)  # Categorical feature
    }
    return pd.DataFrame(num_data)


@pytest.fixture
def missing_value_data():
    """Create a dataset with missing values."""
    num_data = {
        'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],  # Numerical feature with missing value
        'feature2': [5.0, np.nan, 3.0, 2.0, 1.0],  # Numerical feature with missing value
        'category': ['A', None, 'C', 'B', 'A']  # Categorical feature with missing value
    }
    return pd.DataFrame(num_data)


@pytest.mark.parametrize("include_edge_attr", [True, False])
def test_table_to_graph_transformer_fit(dummy_data, fasttext_model_path, include_edge_attr):
    """Test fitting the Table2GraphTransformer with edge attribute variations."""
    transformer = Table2GraphTransformer(
        fasttext_model_path=fasttext_model_path,
        include_edge_attr=include_edge_attr
    )
    transformer.fit(dummy_data)

    # Check that the transformer is fitted and relevant attributes are initialized
    assert transformer.is_fitted_, "Transformer was not properly fitted"
    assert len(transformer.cat_col_names) > 0, "Categorical columns were not identified"
    assert len(transformer.num_col_names) > 0, "Numerical columns were not identified"


@pytest.mark.parametrize("include_edge_attr", [True, False])
def test_table_to_graph_transformer_transform(dummy_data, dummy_labels, fasttext_model_path, include_edge_attr):
    """Test transforming the table data to graph objects."""
    transformer = Table2GraphTransformer(
        fasttext_model_path=fasttext_model_path,
        include_edge_attr=include_edge_attr
    )
    transformer.fit(dummy_data, dummy_labels)

    # Transform the data into graph objects
    graph_data = transformer.transform(dummy_data)

    # Check that the transformation returns a list of Data objects
    assert isinstance(graph_data, list), "Transformation did not return a list"
    assert all(isinstance(g, Data) for g in graph_data), "Transformed data should be torch_geometric Data objects"
    assert len(graph_data) == len(dummy_data), f"Expected {len(dummy_data)} graphs, but got {len(graph_data)}"

    # Check individual graphs for valid data
    for graph in graph_data:
        assert graph.x is not None and graph.x.size(0) > 0, "Node features are missing or empty"
        assert graph.edge_index is not None, "Edge indices are missing"
        if include_edge_attr:
            assert graph.edge_attr is not None, "Edge attributes are missing"
        assert graph.y is not None, "Graph labels are missing"


#def test_edge_index_creation():
#    """Test edge index creation for graph objects."""
    # num_nodes = 5
    # edge_attr = torch.rand((2 * (num_nodes - 1), 4))  # Adjusted size to match edge index requirements
    # edge_index, edge_attr_out = _create_edge_index(num_nodes, edge_attr)

    # # Check that the edge index and edge attributes are returned correctly
    # assert edge_index.shape[1] > 0, "Edge index should not be empty"
    # assert edge_attr_out.shape[0] == edge_index.shape[1], "Edge attribute dimensions mismatch"


def test_large_dataset_transformation(large_dummy_data, fasttext_model_path):
    """Test transformer with a large dataset."""
    transformer = Table2GraphTransformer(fasttext_model_path=fasttext_model_path)
    transformer.fit(large_dummy_data)

    graph_data = transformer.transform(large_dummy_data)

    # Ensure correct number of graphs is generated
    assert len(graph_data) == len(large_dummy_data), "Mismatch in number of graphs generated"
    for graph in graph_data:
        assert graph.x.size(0) > 0, "Node features should not be empty"
        assert graph.edge_index is not None, "Edge indices should not be None"


def test_missing_values_handling(missing_value_data, fasttext_model_path):
    """Test transformer with missing values in the dataset."""
    transformer = Table2GraphTransformer(fasttext_model_path=fasttext_model_path)
    transformer.fit(missing_value_data)

    graph_data = transformer.transform(missing_value_data)

    # Ensure graphs are created despite missing values
    assert len(graph_data) == len(missing_value_data), "Mismatch in number of graphs generated"
    for graph in graph_data:
        assert graph.x.size(0) > 0, "Node features should not be empty"
