import pytest
import torch
from carte_ai.src.carte_model import (
    _carte_calculate_attention,
    _carte_calculate_multihead_output,
    CARTE_Attention,
    CARTE_Block,
    CARTE_Base,
    CARTE_Pretrain,
    CARTE_NN_Model,
    CARTE_NN_Model_Ablation,
)


@pytest.fixture
def dummy_graph_data():
    """Fixture for generating dummy graph data."""
    num_nodes = 10
    input_dim = 8
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)  # Edge connections
    x = torch.randn((num_nodes, input_dim))  # Node features
    edge_attr = torch.randn((edge_index.size(1), input_dim))  # Edge attributes
    return x, edge_index, edge_attr


@pytest.fixture
def dummy_input_data():
    """Fixture for generating input data for models."""
    num_nodes = 10
    input_dim_x = 8
    input_dim_e = 8
    x = torch.randn((num_nodes, input_dim_x))
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn((edge_index.size(1), input_dim_e))
    head_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    return x, edge_index, edge_attr, head_idx


@pytest.fixture
def dummy_input_obj(dummy_input_data):
    """Fixture for creating a reusable input object."""
    x, edge_index, edge_attr, head_idx = dummy_input_data
    input_obj = type('', (), {})()  # Create a dummy object
    input_obj.x = x
    input_obj.edge_index = edge_index
    input_obj.edge_attr = edge_attr
    input_obj.ptr = torch.arange(0, x.size(0) + 1, 3)
    input_obj.head_idx = head_idx
    return input_obj


def test_carte_calculate_attention(dummy_graph_data):
    """Test the attention calculation function."""
    x, edge_index, edge_attr = dummy_graph_data
    
    # Ensure correct indexing of query, key, and value tensors
    key = x[edge_index[1], :]  # Align key with edge_index[1]
    value = x[edge_index[1], :]  # Align value with edge_index[1]
    query = x  # Query uses all nodes
    
    # Call the function with properly indexed inputs
    output, attention = _carte_calculate_attention(edge_index, query, key, value)
    
    # Assertions for output and attention shapes
    assert output.shape == x.shape, "Output shape mismatch"
    assert attention.shape[0] == edge_index.shape[1], "Attention size mismatch"



@pytest.mark.parametrize("num_heads", [1, 2])
def test_carte_calculate_multihead_output(dummy_graph_data, num_heads):
    """Test multi-head attention calculation."""
    x, edge_index, edge_attr = dummy_graph_data
    
    # Ensure correct indexing for key and value tensors
    key = x[edge_index[1], :]  # Align key with edge_index[1]
    value = x[edge_index[1], :]  # Align value with edge_index[1]
    query = x  # Query uses all nodes
    
    # Call the function with correctly indexed inputs
    output, attention = _carte_calculate_multihead_output(edge_index, query, key, value, num_heads=num_heads)
    
    # Assertions for output and attention shapes
    assert output.shape[0] == x.shape[0], "Output shape mismatch"
    assert attention.shape[0] == edge_index.shape[1] * num_heads, "Attention size mismatch"


def test_carte_attention_layer(dummy_graph_data):
    """Test the CARTE_Attention layer."""
    x, edge_index, edge_attr = dummy_graph_data
    attention_layer = CARTE_Attention(input_dim=x.size(1), output_dim=16, num_heads=2)
    output, edge_attr_out = attention_layer(x, edge_index, edge_attr)
    assert output.shape[1] == 16, "Output dimension mismatch"
    assert edge_attr_out.shape[1] == 16, "Edge attribute shape mismatch"


def test_carte_block(dummy_graph_data):
    """Test the CARTE_Block module."""
    x, edge_index, edge_attr = dummy_graph_data
    block = CARTE_Block(input_dim=x.size(1), ff_dim=16, num_heads=2)
    output, edge_attr_out = block(x, edge_index, edge_attr)
    assert output.shape == x.shape, "Output shape mismatch"
    assert edge_attr_out.shape == edge_attr.shape, "Edge attribute shape mismatch"


def test_carte_base_model(dummy_input_obj):
    """Test the CARTE_Base model."""
    base_model = CARTE_Base(
        input_dim_x=dummy_input_obj.x.size(1),
        input_dim_e=dummy_input_obj.edge_attr.size(1),
        hidden_dim=16,
        num_layers=2,
        ff_dim=32,
    )
    output = base_model(dummy_input_obj.x, dummy_input_obj.edge_index, dummy_input_obj.edge_attr)
    assert output.shape[0] == dummy_input_obj.x.shape[0], "Output shape mismatch"


# def test_carte_pretrain_model(dummy_input_obj):
#     """Test the CARTE_Pretrain model."""
#     pretrain_model = CARTE_Pretrain(
#         input_dim_x=dummy_input_obj.x.size(1),
#         input_dim_e=dummy_input_obj.edge_attr.size(1),
#         hidden_dim=16,
#         num_layers=2,
#         ff_dim=32,
#     )
    
#     # Ensure correct indexing of the input and compatibility with head_idx
#     output = pretrain_model(dummy_input_obj)
    
#     # Update assertion to match the correct expected shape
#     assert output.shape[1] == pretrain_model.pretrain_classifier[0].in_features, "Output dimension mismatch"



def test_carte_nn_model(dummy_input_obj):
    """Test the CARTE_NN_Model."""
    nn_model = CARTE_NN_Model(
        input_dim_x=dummy_input_obj.x.size(1),
        input_dim_e=dummy_input_obj.edge_attr.size(1),
        hidden_dim=16,
        output_dim=4,
        num_layers=2,
        ff_dim=32,
    )
    output = nn_model(dummy_input_obj)
    assert output.shape[1] == 4, "Output dimension mismatch"



