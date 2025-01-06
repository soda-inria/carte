import math
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import softmax


# CARTE - Attention and output calculation
def _carte_calculate_attention(
    edge_index: Tensor, query: Tensor, key: Tensor, value: Tensor
):
    ## Fix to work on cpu and gpu provided by Ayoub Kachkach
    # Calculate the scaled-dot product attention
    attention = torch.sum(torch.mul(query[edge_index[0], :], key), dim=1)
    attention = attention / math.sqrt(query.size(1))
    attention = softmax(attention, edge_index[0])
    
    # Ensure `attention` and `value` have the same dtype
    attention = attention.to(value.dtype)
    
    # Generate the output
    src = torch.mul(attention, value.t()).t()
    
    # Ensure `src` and `query` have the same dtype
    src = src.to(query.dtype)
    
    # Use torch.index_add_ to replace scatter function
    output = torch.zeros_like(query).index_add_(0, edge_index[0], src)
    
    return output, attention


# CARTE - output calculation with multi-head (message passing)
def _carte_calculate_multihead_output(
    edge_index: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int = 1,
    concat: bool = True,
):
    if concat:
        H, C = num_heads, query.size(1) // num_heads
        for i in range(H):
            O, A = _carte_calculate_attention(
                edge_index,
                query[:, i * C : (i + 1) * C],
                key[:, i * C : (i + 1) * C],
                value[:, i * C : (i + 1) * C],
            )
            if i == 0:
                output, attention = O, A
            else:
                output = torch.cat((output, O), dim=1)
                attention = torch.cat((attention, A), dim=0)
    else:
        H, C = num_heads, query.size(1)
        for i in range(H):
            O, A = _carte_calculate_attention(
                edge_index,
                query[:, i * C : (i + 1) * C],
                key[:, i * C : (i + 1) * C],
                value[:, i * C : (i + 1) * C],
            )
            if i == 0:
                output, attention = O, A
            else:
                output = torch.cat((output, O), dim=0)
                attention = torch.cat((attention, A), dim=0)
        output = output / H
        attention = attention / H
    return output, attention


# CARTE - Attention Layer
class CARTE_Attention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 1,
        concat: bool = True,
        read_out: bool = False,
    ):
        super(CARTE_Attention, self).__init__()

        if concat:
            assert output_dim % num_heads == 0
            self.lin_query = nn.Linear(input_dim, output_dim, bias=False)
            self.lin_key = nn.Linear(input_dim, output_dim, bias=False)
            self.lin_value = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.lin_query = nn.Linear(input_dim, num_heads * output_dim, bias=False)
            self.lin_key = nn.Linear(input_dim, num_heads * output_dim, bias=False)
            self.lin_value = nn.Linear(input_dim, num_heads * output_dim, bias=False)

        if not read_out:
            self.lin_edge = nn.Linear(input_dim, output_dim)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.concat = concat
        self.readout = read_out

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_query.reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_value.reset_parameters()
        if not self.readout:
            self.lin_edge.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        return_attention: bool = False,
    ):
        Z = torch.mul(edge_attr, x[edge_index[1]])

        query = self.lin_query(x)
        key = self.lin_key(Z)
        value = self.lin_value(Z)

        output, attention = _carte_calculate_multihead_output(
            edge_index=edge_index,
            query=query,
            key=key,
            value=value,
            num_heads=self.num_heads,
            concat=self.concat,
        )

        if not self.readout:
            edge_attr = self.lin_edge(edge_attr)

        if return_attention:
            return output, edge_attr, attention
        else:
            return output, edge_attr


# CARTE - single encoding block
class CARTE_Block(nn.Module):
    def __init__(
        self,
        input_dim: int,
        ff_dim: int,
        num_heads: int = 1,
        concat: bool = True,
        dropout: float = 0.1,
        read_out: bool = False,
    ):
        super().__init__()

        # Graph Attention Layer
        self.g_attn = CARTE_Attention(
            input_dim, input_dim, num_heads, concat, read_out=read_out
        )

        # Two-layer MLP + Layers to apply in between the main layers for x and edges
        self.linear_net_x = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(ff_dim, input_dim),
        )
        self.norm1_x = nn.LayerNorm(input_dim)
        self.norm2_x = nn.LayerNorm(input_dim)

        self.read_out = read_out
        if not self.read_out:
            self.linear_net_e = nn.Sequential(
                nn.Linear(input_dim, ff_dim),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(ff_dim, input_dim),
            )
            self.norm1_e = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ):
        # Attention part
        attn_out_x, attn_out_e = self.g_attn(x, edge_index, edge_attr)
        x = self.dropout(attn_out_x)
        x = self.norm1_x(x)

        # MLP part - Node
        linear_out_x = self.linear_net_x(x)
        x = self.dropout(linear_out_x)
        x = self.norm2_x(x)

        # MLP part - Edge
        if not self.read_out:
            edge_attr = self.linear_net_e(attn_out_e)
            edge_attr = edge_attr + self.dropout(edge_attr)
            edge_attr = self.norm1_e(edge_attr)
            return x, edge_attr
        else:
            return x


# CARTE - contrast block
class CARTE_Contrast(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        x = nn.functional.normalize(x, dim=1)

        # Cosine similarity
        x = 1 - (torch.cdist(x, x) / 2)

        return x


# CARTE - finetune base block
class CARTE_Base(nn.Module):
    def __init__(
        self,
        input_dim_x: int,
        input_dim_e: int,
        hidden_dim: int,
        num_layers: int,
        **block_args,
    ):
        super(CARTE_Base, self).__init__()

        self.initial_x = nn.Sequential(
            nn.Linear(input_dim_x, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.initial_e = nn.Sequential(
            nn.Linear(input_dim_e, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.layers = nn.ModuleList(
            [CARTE_Block(input_dim=hidden_dim, **block_args) for _ in range(num_layers)]
        )

        self.read_out_block = CARTE_Block(
            input_dim=hidden_dim, read_out=True, **block_args
        )

    def forward(self, x, edge_index, edge_attr, return_attention=False):
        # Initial layer for the node/edge features
        x = self.initial_x(x)
        edge_attr = self.initial_e(edge_attr)

        for l in self.layers:
            x, edge_attr = l(x, edge_index, edge_attr)

        x = self.read_out_block(x, edge_index, edge_attr)

        if return_attention:
            attention_maps = []
            for l in self.layers:
                _, _, attention = l.g_attn(x, edge_index, edge_attr, return_attention)
                attention_maps.append(attention)
            return x, attention_maps
        elif not return_attention:
            return x


# CARTE - Pretrain Model
class CARTE_Pretrain(nn.Module):
    def __init__(
        self,
        input_dim_x: int,
        input_dim_e: int,
        hidden_dim: int,
        num_layers: int,
        **block_args,
    ):
        super(CARTE_Pretrain, self).__init__()

        self.ft_base = CARTE_Base(
            input_dim_x=input_dim_x,
            input_dim_e=input_dim_e,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **block_args,
        )

        self.pretrain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, elementwise_affine=False),
            CARTE_Contrast(),
        )

    def forward(self, input):
        x, edge_index, edge_attr, head_idx = (
            input.x.clone(),
            input.edge_index,
            input.edge_attr.clone(),
            input.head_idx,
        )

        x = self.ft_base(x, edge_index, edge_attr)
        x = x[head_idx, :]
        x = self.pretrain_classifier(x)

        return x


# CARTE - Downstream Model
class CARTE_NN_Model(nn.Module):
    def __init__(
        self,
        input_dim_x: int,
        input_dim_e: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        **block_args,
    ):
        super(CARTE_NN_Model, self).__init__()

        self.ft_base = CARTE_Base(
            input_dim_x=input_dim_x,
            input_dim_e=input_dim_e,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **block_args,
        )

        self.ft_classifier = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim / 2)),
            nn.ReLU(),
            nn.LayerNorm(int(hidden_dim / 2)),
            nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4)),
            nn.ReLU(),
            nn.LayerNorm(int(hidden_dim / 4)),
            nn.Linear(int(hidden_dim / 4), output_dim),
        )

    def forward(self, input):
        x, edge_index, edge_attr, head_idx = (
            input.x.clone(),
            input.edge_index.clone(),
            input.edge_attr.clone(),
            input.ptr[:-1],
        )

        x = self.ft_base(x, edge_index, edge_attr)
        x = x[head_idx, :]
        x = self.ft_classifier(x)

        return x


# CARTE - Downstream Ablation model
class CARTE_NN_Model_Ablation(nn.Module):
    def __init__(
        self,
        ablation_method: str,
        input_dim_x: int,
        input_dim_e: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        **block_args,
    ):
        super(CARTE_NN_Model_Ablation, self).__init__()

        self.ablation_method = ablation_method

        self.ft_base = CARTE_Base(
            input_dim_x=input_dim_x,
            input_dim_e=input_dim_e,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **block_args,
        )

        self.ft_classifier = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim / 2)),
            nn.ReLU(),
            nn.LayerNorm(int(hidden_dim / 2)),
            nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4)),
            nn.ReLU(),
            nn.LayerNorm(int(hidden_dim / 4)),
            nn.Linear(int(hidden_dim / 4), output_dim),
        )

    def forward(self, input):
        x, edge_index, edge_attr, head_idx = (
            input.x.clone(),
            input.edge_index.clone(),
            input.edge_attr.clone(),
            input.ptr[:-1],
        )

        if "exclude-attention" not in self.ablation_method:
            x = self.ft_base(x, edge_index, edge_attr)
        x = x[head_idx, :]
        x = self.ft_classifier(x)

        return x
