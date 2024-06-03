"""
Graphlet construction for knowledge graph data.
"""

import numpy as np
import math
import torch
from typing import List, Union
from torch import Tensor
from torch_geometric.data import Data

## K-hop Subgraph Extraction
def _k_hop_subgraph(
    node_index: int,
    num_hops: int,
    max_nodes: int,
    edge_index: Tensor,
    edge_type: Tensor,
    value_type_mask: Tensor,
):

    num_nodes = edge_index.max().item() + 1
    head, tail = edge_index

    node_mask = head.new_empty(num_nodes, dtype=torch.bool)
    reduce_mask_ = head.new_empty(edge_index.size(1), dtype=torch.bool)
    reduce_mask_.fill_(False)
    subset = int(node_index)
    limit = [int(math.ceil(max_nodes * 10 ** (-i))) for i in range(num_hops)]

    for i in range(num_hops):
        node_mask.fill_(False)
        node_mask[subset] = True
        idx_rm = node_mask[head].nonzero().view(-1)
        idx_rm = idx_rm[torch.randperm(idx_rm.size(0))[: limit[i]]]
        reduce_mask_[idx_rm] = True
        subset = tail[reduce_mask_].unique()

    edge_index_ = edge_index[:, reduce_mask_].clone()
    edge_type_ = edge_type[reduce_mask_].clone()
    value_type_mask_ = value_type_mask[reduce_mask_].clone()

    subset = edge_index_.unique()

    mapping = torch.reshape(torch.tensor((node_index, 0)), (2, 1))
    mapping_temp = torch.vstack(
        (subset[subset != node_index], torch.arange(1, subset.size()[0]))
    )
    mapping = torch.hstack((mapping, mapping_temp))

    head_ = edge_index_[0, :]
    tail_ = edge_index_[1, :]

    sort_idx = torch.argsort(mapping[0, :])
    idx_h = torch.searchsorted(mapping[0, :], head_, sorter=sort_idx)
    idx_t = torch.searchsorted(mapping[0, :], tail_, sorter=sort_idx)

    out_h = mapping[1, :][sort_idx][idx_h]
    out_t = mapping[1, :][sort_idx][idx_t]

    edge_index_new = torch.vstack((out_h, out_t))

    edge_index_new = torch.hstack(
        (
            edge_index_new,
            torch.tensor(
                (
                    [0],
                    [edge_index_new.max().item() + 1],
                )
            ),
        )
    )
    edge_type_new = torch.hstack((edge_type_, torch.zeros(1, dtype=torch.long)))
    value_type_mask_new = torch.hstack(
        (value_type_mask_, torch.zeros(1, dtype=torch.long))
    )

    mapping = torch.hstack(
        (mapping, torch.tensor([[node_index], [edge_index_new.max().item()]]))
    )

    return edge_index_new, edge_type_new, value_type_mask_new, mapping


## Subgraph with assigned nodes
def _subgraph(subset: Tensor, edge_index: Tensor):

    device = edge_index.device
    num_nodes = edge_index.max().item() + 1

    if isinstance(subset, (list, tuple)):
        subset = torch.tensor(subset, dtype=torch.long, device=device)

    subset_ = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    subset_[subset] = True
    subset = subset_

    node_mask = subset
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]

    mapping = torch.vstack((edge_index.unique(), torch.argsort(edge_index.unique())))

    head_ = edge_index[0, :]
    tail_ = edge_index[1, :]

    sort_idx = torch.argsort(mapping[0, :])
    idx_h = torch.searchsorted(mapping[0, :], head_, sorter=sort_idx)
    idx_t = torch.searchsorted(mapping[0, :], tail_, sorter=sort_idx)

    out_h = mapping[1, :][sort_idx][idx_h]
    out_t = mapping[1, :][sort_idx][idx_t]

    edge_list_new = torch.vstack((out_h, out_t))

    return edge_list_new, edge_mask, mapping


## Graph perturbation with truncation
def _perturb_truncate_node(data, per_keep: float, keep_name: bool = True):
    data_perturb = data.clone()
    idx_map = data.edge_index[1, (data.edge_index[0] == 0)]
    idx_map, _ = torch.sort(idx_map)
    idx_keep = torch.zeros(1, dtype=torch.long)
    idx_keep_ = torch.ones(idx_map.size(0), dtype=bool)
    if keep_name:
        idx_keep = torch.hstack((idx_keep, idx_map[-1]))
        idx_keep_[-1] = False
    idx_keep_ = idx_map[idx_keep_.nonzero().view(-1)]
    if idx_keep_.size(0) > 1:
        num_keep = torch.randint(
            math.floor(per_keep * idx_keep_.size(0)), idx_keep_.size(0), (1,)
        ).item()
        # num_keep = math.floor(per_keep * idx_keep_.size(0)) + 1
        idx_keep_ = torch.tensor(np.random.choice(idx_keep_, num_keep, replace=False))
        idx_keep = torch.hstack((idx_keep, idx_keep_))
    edge_index, edge_mask, mask_ = _subgraph(idx_keep, data_perturb.edge_index)
    data_perturb.edge_index = edge_index
    data_perturb.edge_type = data_perturb.edge_type[edge_mask]
    data_perturb.edge_attr = data_perturb.edge_attr[edge_mask, :]
    data_perturb.x = data_perturb.x[mask_[1, :]]
    Z = torch.mul(data_perturb.edge_attr, data_perturb.x[data_perturb.edge_index[1]])
    data_perturb.x[0, :] = Z[(data_perturb.edge_index[0] == 0), :].mean(dim=0)
    return data_perturb

## Remove duplicate function
def _remove_duplicates(
    edge_index: Tensor,
    edge_type: Tensor = None,
    edge_attr: Tensor = None,
    perturb_tensor: Tensor = None,
):
    nnz = edge_index.size(1)
    num_nodes = edge_index.max().item() + 1

    idx = edge_index.new_empty(nnz + 1)
    idx[0] = -1
    idx[1:] = edge_index[0]
    idx[1:].mul_(num_nodes).add_(edge_index[1])

    if edge_type is not None:
        idx[1:].add_((edge_type + 1) * (10 ** (len(str(num_nodes)) + 3)))

    idx[1:], perm = torch.sort(
        idx[1:],
    )

    mask = idx[1:] > idx[:-1]

    edge_index = edge_index[:, perm]
    edge_index = edge_index[:, mask]

    if edge_type is not None:
        edge_type, edge_attr = edge_type[perm], edge_attr[perm, :]
        edge_type, edge_attr = edge_type[mask], edge_attr[mask, :]
        if perturb_tensor is not None:
            perturb_tensor = perturb_tensor[perm]
            perturb_tensor = perturb_tensor[mask]
            return edge_index, edge_type, edge_attr, perturb_tensor
        else:
            return edge_index, edge_type, edge_attr
    else:
        return edge_index


## To undirected function
def _to_undirected(
    edge_index: Tensor,
    edge_type: Tensor = None,
    edge_attr: Tensor = None,
    idx_perturb=None,
):
    row = torch.cat([edge_index[0, :], edge_index[1, :]])
    col = torch.cat([edge_index[1, :], edge_index[0, :]])

    edge_index = torch.stack([row, col], dim=0)

    if edge_type is not None:
        edge_type = torch.cat([edge_type, edge_type])
        edge_attr = torch.vstack((edge_attr, edge_attr))
        if idx_perturb is not None:
            perturb_tensor = torch.zeros(edge_type.size(0))
            perturb_tensor[idx_perturb] = -1
            perturb_tensor = torch.cat([perturb_tensor, perturb_tensor])
            edge_index, edge_type, edge_attr, perturb_tensor = _remove_duplicates(
                edge_index=edge_index,
                edge_type=edge_type,
                edge_attr=edge_attr,
                perturb_tensor=perturb_tensor,
            )
            idx_perturb = (perturb_tensor < 0).nonzero().squeeze()
            return edge_index, edge_type, edge_attr, idx_perturb
        else:
            edge_index, edge_type, edge_attr = _remove_duplicates(
                edge_index=edge_index,
                edge_type=edge_type,
                edge_attr=edge_attr,
            )
        idx_perturb = []
        return edge_index, edge_type, edge_attr, idx_perturb
    else:
        edge_index = _remove_duplicates(edge_index=edge_index)
        return edge_index



# Graphlet class to construct a graphlet of a given entity
class Graphlet:
    def __init__(
        self,
        data_kg,
        num_hops: int = 1,
        max_nodes: int = 15,
    ):
        super(Graphlet, self).__init__()

        self.data_kg = data_kg
        self.num_hops = num_hops
        self.max_nodes = max_nodes

        self.x_total = data_kg["x_total"]
        self.edge_index = data_kg["edge_index"]
        self.edge_attr_total = data_kg["edge_attr_total"]
        self.edge_type = data_kg["edge_type"]
        self.ent2idx = data_kg["ent2idx"]
        self.rel2idx = data_kg["rel2idx"]
        self.value_type_mask = data_kg["value_type_mask"]

    def make_batch(
        self,
        center_indices: Union[int, List[int], Tensor],
        num_perturb: int = 0,
        fraction_perturb: float = 0.9,
        undirected: bool = True,
    ):
        if isinstance(center_indices, Tensor):
            center_indices = center_indices.tolist()
        if isinstance(center_indices, int):
            center_indices = [center_indices]

        # Obtain the of entities and edge_types in the batch (reduced set)
        head_ = self.edge_index[0, :]
        tail_ = self.edge_index[1, :]

        node_mask = head_.new_empty(self.edge_index.max().item() + 1, dtype=torch.bool)
        node_mask.fill_(False)

        subset = center_indices

        for _ in range(self.num_hops):
            node_mask[subset] = True
            reduce_mask = node_mask[head_]
            subset = tail_[reduce_mask].unique()

        edge_index_reduced = self.edge_index[:, reduce_mask]
        edge_type_reduced = self.edge_type[reduce_mask]
        value_type_mask_reduced = self.value_type_mask[reduce_mask]

        # Obtain the list of data with original and perturbed graphs
        data_total = []
        data_perturb_ = []
        data_original_ = [
            self._make_graphlet(
                node_index,
                edge_index_reduced,
                edge_type_reduced,
                value_type_mask_reduced,
                undirected,
            )
            for node_index in center_indices
        ]

        if num_perturb != 0:
            for data in data_original_:
                per_keep = 1 - fraction_perturb
                data_perturb_ += [
                    _perturb_truncate_node(
                        data=data, per_keep=per_keep, keep_name=True
                    )
                    for _ in range(num_perturb)
                ]

        data_total += data_original_ + data_perturb_



        return data_total

    def _make_graphlet(
        self,
        node_index: int,
        edge_index: Tensor,
        edge_type: Tensor,
        value_type_mask: Tensor,
        undirected: bool,
    ):

        edge_index, edge_type, value_type_mask, mapping = _k_hop_subgraph(
            edge_index=edge_index,
            node_index=node_index,
            max_nodes=self.max_nodes,
            num_hops=self.num_hops,
            edge_type=edge_type,
            value_type_mask=value_type_mask,
        )

        edge_attr = self.edge_attr_total[edge_type, :]
        x = self.x_total[mapping[0, :], :]

        if value_type_mask.sum().item() > 0:
            x[edge_index[1, (value_type_mask == 1)]] = torch.mul(
                x[edge_index[1, (value_type_mask == 1)]],
                edge_attr[(value_type_mask == 1), :],
            )

        Z = torch.mul(edge_attr, x[edge_index[1]])
        x[0, :] = Z[(edge_index[0] == 0), :].mean(dim=0)

        if undirected:
            edge_index, edge_type, edge_attr, _ = _to_undirected(
                edge_index,
                edge_type,
                edge_attr,
            )
        else:
            pass

        data_out = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_attr=edge_attr,
            g_idx=node_index,
            y=torch.tensor([1]),
            flag_perturb=torch.tensor([0]),
            mapping=torch.transpose(mapping, 0, 1),
        )

        return data_out
