import pytest
import torch
from carte_ai.src.carte_model import (
    _carte_calculate_attention,
    _carte_calculate_multihead_output,
    CARTE_Attention,
    CARTE_Block,
    CARTE_Base,
    CARTE_NN_Model,
    CARTE_NN_Model_Ablation
)

@pytest.fixture
def dummy_graph_data():
    """Create dummy graph data for testing."""
    num_nodes = 10
    input_dim = 8
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)  # Edge connections
    x = torch.randn((num_nodes, input_dim))  # Node features
    edge_attr = torch.randn((edge_index.size(1), input_dim))  # Edge attributes
    return x, edge_index, edge_attr

@pytest.fixture
def dummy_input_data():
    """Create dummy input data for NN models."""
    num_nodes = 10
    input_dim_x = 8
    input_dim_e = 8
    x = torch.randn((num_nodes, input_dim_x))
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn((edge_index.size(1), input_dim_e))
    head_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    return x, edge_index, edge_attr, head_idx

def test_carte_calculate_attention(dummy_graph_data):
    """Test single-head attention calculation."""
    x, edge_index, edge_attr = dummy_graph_data
    output, attention = _carte_calculate_attention(edge_index, x, x, x)
    
    assert output.shape[0] == x.shape[0], "Output size mismatch"
    assert attention.shape[0] == edge_index.shape[1], "Attention size mismatch"

@pytest.mark.parametrize("num_heads", [1, 2])
def test_carte_calculate_multihead_output(dummy_graph_data, num_heads):
    """Test multi-head attention calculation."""
    x, edge_index, edge_attr = dummy_graph_data
    output, attention = _carte_calculate_multihead_output(edge_index, x, x, x, num_heads=num_heads)
    
    assert output.shape[0] == x.shape[0], "Output size mismatch"
    assert attention.shape[0] == edge_index.shape[1] * num_heads, "Attention size mismatch"

def test_carte_attention_layer(dummy_graph_data):
    """Test CARTE_Attention layer."""
    x, edge_index, edge_attr = dummy_graph_data
    attention_layer = CARTE_Attention(input_dim=x.size(1), output_dim=16, num_heads=2)
    
    output, edge_attr_out = attention_layer(x, edge_index, edge_attr)
    
    assert output.shape[1] == 16, "Output dimension mismatch"
    assert edge_attr_out.shape == edge_attr.shape, "Edge attribute shape mismatch"

def test_carte_block(dummy_graph_data):
    """Test CARTE_Block layer."""
    x, edge_index, edge_attr = dummy_graph_data
    block = CARTE_Block(input_dim=x.size(1), ff_dim=16, num_heads=2)
    
    output, edge_attr_out = block(x, edge_index, edge_attr)
    
    assert output.shape == x.shape, "Output shape mismatch"
    assert edge_attr_out.shape == edge_attr.shape, "Edge attribute shape mismatch"

def test_carte_base_model(dummy_input_data):
    """Test the CARTE_Base model."""
    x, edge_index, edge_attr, head_idx = dummy_input_data
    base_model = CARTE_Base(input_dim_x=x.size(1), input_dim_e=edge_attr.size(1), hidden_dim=16, num_layers=2)
    
    output = base_model(x, edge_index, edge_attr)
    
    assert output.shape[0] == x.shape[0], "Output shape mismatch"

def test_carte_nn_model(dummy_input_data):
    """Test the CARTE_NN_Model."""
    x, edge_index, edge_attr, head_idx = dummy_input_data
    nn_model = CARTE_NN_Model(input_dim_x=x.size(1), input_dim_e=edge_attr.size(1), hidden_dim=16, output_dim=4, num_layers=2)
    
    input_obj = type('', (), {})()  # Create dummy object for input
    input_obj.x = x
    input_obj.edge_index = edge_index
    input_obj.edge_attr = edge_attr
    input_obj.ptr = torch.arange(0, x.size(0) + 1, 3)
    
    output = nn_model(input_obj)
    
    assert output.shape[1] == 4, "Output dimension mismatch"
    assert output.shape[0] == input_obj.ptr.size(0) - 1, "Head index mismatch"

@pytest.mark.parametrize("ablation_method", ["exclude-attention", "exclude-attention-edge"])
def test_carte_ablation_model(dummy_input_data, ablation_method):
    """Test CARTE_NN_Model_Ablation."""
    x, edge_index, edge_attr, head_idx = dummy_input_data
    nn_model_ablation = CARTE_NN_Model_Ablation(
        ablation_method=ablation_method,
        input_dim_x=x.size(1),
        input_dim_e=edge_attr.size(1),
        hidden_dim=16,
        output_dim=4,
        num_layers=2,
    )
    
    input_obj = type('', (), {})()  # Create dummy object for input
    input_obj.x = x
    input_obj.edge_index = edge_index
    input_obj.edge_attr = edge_attr
    input_obj.ptr = torch.arange(0, x.size(0) + 1, 3)
    
    output = nn_model_ablation(input_obj)
    
    assert output.shape[1] == 4, "Output dimension mismatch for ablation model"
    assert output.shape[0] == input_obj.ptr.size(0) - 1, "Head index mismatch for ablation model"
