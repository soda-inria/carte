import pytest
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from carte_ai.src.carte_table_to_graph import Table2GraphTransformer, _create_edge_index

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

@pytest.mark.parametrize("include_edge_attr", [True, False])
def test_table_to_graph_transformer_fit(dummy_data, include_edge_attr):
    """Test fitting the Table2GraphTransformer."""
    transformer = Table2GraphTransformer(include_edge_attr=include_edge_attr, lm_model="fasttext", fasttext_model_path="path/to/fasttext.bin")
    transformer.fit(dummy_data)
    
    # Check that the transformer is fitted and relevant attributes are initialized
    assert transformer.is_fitted_, "Transformer was not properly fitted"
    assert len(transformer.cat_col_names) > 0, "Categorical columns were not identified"
    assert len(transformer.num_col_names) > 0, "Numerical columns were not identified"

@pytest.mark.parametrize("include_edge_attr", [True, False])
def test_table_to_graph_transformer_transform(dummy_data, dummy_labels, include_edge_attr):
    """Test transforming the table data to graph objects."""
    transformer = Table2GraphTransformer(include_edge_attr=include_edge_attr, lm_model="fasttext", fasttext_model_path="path/to/fasttext.bin")
    transformer.fit(dummy_data, dummy_labels)
    
    # Transform the data into graph objects
    graph_data = transformer.transform(dummy_data)
    
    # Check that the transformation returns a list of Data objects
    assert isinstance(graph_data, list), "Transformation did not return a list"
    assert all(isinstance(g, Data) for g in graph_data), "Transformed data should be torch_geometric Data objects"
    assert len(graph_data) == len(dummy_data), f"Expected {len(dummy_data)} graphs, but got {len(graph_data)}"

def test_edge_index_creation():
    """Test edge index creation for graph objects."""
    num_nodes = 5
    edge_attr = torch.rand((num_nodes - 1, 4))  # Edge attributes for testing
    edge_index, edge_attr_out = _create_edge_index(num_nodes, edge_attr)
    
    # Check that the edge index and edge attributes are returned correctly
    assert edge_index.shape[1] > 0, "Edge index should not be empty"
    assert edge_attr_out.shape[0] > 0, "Edge attributes should not be empty"
    assert edge_attr_out.shape == (edge_index.shape[1], edge_attr.shape[1]), "Edge attribute dimensions mismatch"

def test_invalid_fasttext_model_path(dummy_data):
    """Test that the transformer raises an error for invalid FastText model path."""
    with pytest.raises(ValueError):
        transformer = Table2GraphTransformer(lm_model="fasttext")
        transformer.fit(dummy_data)

@pytest.mark.parametrize("lm_model", ["fasttext", "minhash"])
def test_table_to_graph_with_different_lm_models(dummy_data, lm_model):
    """Test Table2GraphTransformer with different language models."""
    transformer = Table2GraphTransformer(lm_model=lm_model, n_components=100, fasttext_model_path="path/to/fasttext.bin")
    transformer.fit(dummy_data)
    
    # Ensure the language model is loaded
    assert transformer.lm_model_ is not None, f"Language model {lm_model} was not loaded properly"

@pytest.mark.parametrize("num_nodes, self_loop", [(3, True), (3, False), (5, True)])
def test_create_edge_index(num_nodes, self_loop):
    """Test creating edge indices with self-loops and without."""
    edge_attr = torch.rand((num_nodes - 1, 4))
    edge_index, edge_attr_out = _create_edge_index(num_nodes, edge_attr, self_loop=self_loop)
    
    if self_loop:
        assert edge_index.shape[1] > num_nodes, "Expected self-loops, but none were added"
    else:
        assert edge_index.shape[1] == 2 * (num_nodes - 1), "Expected no self-loops, but they were added"

@pytest.mark.parametrize("cat_only", [True, False])
def test_table_to_graph_cat_vs_num(dummy_data, cat_only):
    """Test the transformer with either categorical-only or numerical-only data."""
    if cat_only:
        dummy_data = dummy_data.drop(columns=["feature1", "feature2"])  # Only keep categorical columns
    else:
        dummy_data = dummy_data.drop(columns=["category"])  # Only keep numerical columns
    
    transformer = Table2GraphTransformer(lm_model="fasttext", fasttext_model_path="path/to/fasttext.bin")
    transformer.fit(dummy_data)
    
    graph_data = transformer.transform(dummy_data)
    
    # Ensure we have graphs
    assert len(graph_data) == len(dummy_data), "Mismatch in number of graphs generated"

