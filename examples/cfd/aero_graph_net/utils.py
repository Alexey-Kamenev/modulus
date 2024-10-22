# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Literal, Mapping, Optional

import numpy as np

import torch
from torch import Tensor

from dgl import DGLGraph

from modulus.distributed import DistributedManager
from modulus.models.gnn_layers import (
    CuGraphCSC,
    partition_graph_nodewise,
    partition_graph_by_coordinate_bbox,
)


class RRMSELoss(torch.nn.Module):
    """Relative RMSE loss."""

    def forward(self, pred: Tensor, target: Tensor):
        return (
            torch.linalg.vector_norm(pred - target) / torch.linalg.vector_norm(target)
        ).mean()


def batch_as_dict(
    batch, device: Optional[torch.device | str] = None
) -> Mapping[str, Any]:
    """Wraps provided batch in a dictionary, if needed.

    If `device` is not None, moves all Tensor items to the device.
    """

    batch = batch if isinstance(batch, Mapping) else {"graph": batch}
    if device is None:
        return batch
    return {
        k: v.to(device) if isinstance(v, (Tensor, DGLGraph)) else v
        for k, v in batch.items()
    }


def relative_lp_error(pred, y, p=2):
    """
    Calculate relative L2 error norm
    Parameters:
    -----------
    pred: torch.Tensor
        Prediction
    y: torch.Tensor
        Ground truth
    Returns:
    --------
    error: float
        Calculated relative L2 error norm (percentage) on cpu
    """

    error = (
        torch.mean(torch.linalg.norm(pred - y, ord=p) / torch.linalg.norm(y, ord=p))
        .cpu()
        .numpy()
    )
    return error * 100


def partition_graph(
    graph: DGLGraph,
    partition_type: Literal["nodewise", "coordinate_bbox"],
    partition_group_name: str,
) -> tuple[CuGraphCSC, Tensor, Tensor, Tensor]:
    """
    Partitions graph across multiple GPUs.

    Parameters:
    -----------
    graph: DGLGraph
        Graph to partition.
    partition_type: str
        Partition type.
    partition_group_name: str
        Partition group name.
    Returns:
    --------
    """

    dist = DistributedManager()

    offsets, indices, edge_perm = graph.adj_tensors("csc")
    match partition_type:
        case "coordinate_bbox":
            # Split pos into slices along x axis.
            pos = graph.ndata["pos"]
            pos_min = pos.min(axis=0)[0].cpu().numpy()
            pos_max = pos.max(axis=0)[0].cpu().numpy()
            num_slices = dist.world_size
            x_steps = np.linspace(
                pos_min[0],
                pos_max[0],
                num_slices + 1,
                endpoint=True,
                dtype=np.float32,
            )
            min_max_slices = np.zeros((2, num_slices, 3), dtype=np.float32)
            # min coords
            min_max_slices[0, :, 0] = x_steps[:-1]
            min_max_slices[0, :, 1:] = pos_min[1:]
            # max coords
            min_max_slices[1, :, 0] = x_steps[1:]
            min_max_slices[1, :, 1:] = pos_max[1:]
            graph_partition = partition_graph_by_coordinate_bbox(
                offsets.to(dtype=torch.int64),
                indices.to(dtype=torch.int64),
                pos,
                pos,
                min_max_slices[0],
                min_max_slices[1],
                dist.world_size,
                dist.rank,
                dist.device,
            )
        case "nodewise":
            graph_partition = partition_graph_nodewise(
                offsets.to(dtype=torch.int64),
                indices.to(dtype=torch.int64),
                dist.world_size,
                dist.rank,
                dist.device,
            )
        case other:
            assert False, f"Unsupported partition type {other}"

    graph_multi_gpu = CuGraphCSC(
        offsets.to(dist.device),
        indices.to(dist.device),
        graph.num_src_nodes(),
        graph.num_dst_nodes(),
        partition_size=dist.world_size,
        partition_group_name=partition_group_name,
        graph_partition=graph_partition,
    )
    node_feats = graph_multi_gpu.get_dst_node_features_in_partition(
        graph.ndata["x"].to(dist.device)
    )
    edge_feats = graph.edata["x"][edge_perm]
    edge_feats = graph_multi_gpu.get_edge_features_in_partition(
        edge_feats.to(dist.device)
    )
    y = graph_multi_gpu.get_dst_node_features_in_partition(
        graph.ndata["y"].to(dist.device)
    )
    return graph_multi_gpu, node_feats, edge_feats, y
