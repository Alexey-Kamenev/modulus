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

import os

import dgl
import pytest
import torch
from models.common import validate_forward_accuracy
from pytest_utils import import_or_fail

from modulus.distributed import DistributedManager
from modulus.models.gnn_layers import CuGraphCSC, partition_graph_nodewise
from modulus.models.meshgraphnet.bsms_mgn import BiStrideMeshGraphNet


@pytest.fixture
def ahmed_data_dir():
    path = "/data/nfs/modulus-data/datasets/ahmed_body/"
    return path


@import_or_fail("sparse_dot_mkl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_bsms_mgn_forward(pytestconfig, device):
    from modulus.datapipes.gnn.bsms import BistrideMultiLayerGraphDataset

    torch.manual_seed(1)

    # Create a simple graph.
    num_nodes = 8
    edges = (
        torch.arange(num_nodes - 1),
        torch.arange(num_nodes - 1) + 1,
    )
    pos = torch.randn((num_nodes, 3))

    graph = dgl.graph(edges)
    graph = dgl.to_bidirected(graph)

    num_layers = 2
    input_dim_nodes = 10
    input_dim_edges = 4
    output_dim = 4

    graph.ndata["pos"] = pos
    graph.ndata["x"] = torch.randn(num_nodes, input_dim_nodes)
    graph.edata["x"] = torch.randn(graph.num_edges(), input_dim_edges)

    dataset = BistrideMultiLayerGraphDataset([graph], num_layers)
    assert len(dataset) == 1

    # Create a model.
    model = BiStrideMeshGraphNet(
        input_dim_nodes=input_dim_nodes,
        input_dim_edges=input_dim_edges,
        output_dim=output_dim,
        num_layers_bistride=num_layers,
        processor_size=2,
        hidden_dim_processor=32,
        hidden_dim_node_encoder=16,
        hidden_dim_edge_encoder=16,
    ).to(device)
    model.eval()

    s0 = dataset[0]
    g0 = s0["graph"].to(device)
    ms_edges0 = s0["ms_edges"]
    ms_ids0 = s0["ms_ids"]
    node_features = g0.ndata["x"]
    edge_features = g0.edata["x"]
    pred = model(node_features, edge_features, g0, ms_edges0, ms_ids0)

    # Check output shape.
    assert pred.shape == (g0.num_nodes(), output_dim)

    assert validate_forward_accuracy(
        model,
        (node_features, edge_features, g0, ms_edges0, ms_ids0),
    )


@import_or_fail("sparse_dot_mkl")
def test_bsms_mgn_ahmed(pytestconfig, ahmed_data_dir):
    from modulus.datapipes.gnn.ahmed_body_dataset import AhmedBodyDataset
    from modulus.datapipes.gnn.bsms import BistrideMultiLayerGraphDataset

    device = torch.device("cuda:0")

    torch.manual_seed(1)

    # Construct multi-scale dataset out of standard Ahmed Body dataset.
    ahmed_dataset = AhmedBodyDataset(
        data_dir=ahmed_data_dir,
        split="train",
        num_samples=2,
    )

    num_levels = 2
    dataset = BistrideMultiLayerGraphDataset(ahmed_dataset, num_levels)

    output_dim = 4
    # Construct model.
    model = BiStrideMeshGraphNet(
        input_dim_nodes=11,
        input_dim_edges=4,
        output_dim=output_dim,
        processor_size=2,
        hidden_dim_processor=32,
        hidden_dim_node_encoder=16,
        hidden_dim_edge_encoder=16,
    ).to(device)

    s0 = dataset[0]
    g0 = s0["graph"].to(device)
    ms_edges0 = s0["ms_edges"]
    ms_ids0 = s0["ms_ids"]
    pred = model(g0.ndata["x"], g0.edata["x"], g0, ms_edges0, ms_ids0)

    # Check output shape.
    assert pred.shape == (g0.num_nodes(), output_dim)


def distributed_setup(rank, model_parallel_size, pg_name):
    os.environ["RANK"] = f"{rank}"
    os.environ["WORLD_SIZE"] = f"{model_parallel_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12355)
    DistributedManager._shared_state = {}

    DistributedManager.initialize()

    DistributedManager.create_process_subgroup(
        name=pg_name,
        size=model_parallel_size,
    )


def run_bsms_partitioning(rank, model_parallel_size):
    from modulus.datapipes.gnn.bsms import BistrideMultiLayerGraphDataset

    torch.manual_seed(1)

    pg_name = "graph_partition"
    distributed_setup(rank, model_parallel_size, pg_name)

    device = torch.device("cuda:0")

    # Create a simple graph.
    num_nodes = 8
    edges = (
        torch.arange(num_nodes - 1),
        torch.arange(num_nodes - 1) + 1,
    )
    pos = torch.randn((num_nodes, 3))

    graph = dgl.graph(edges)
    graph = dgl.to_bidirected(graph)

    num_layers = 2
    input_dim_nodes = 10
    input_dim_edges = 4

    graph.ndata["pos"] = pos
    graph.ndata["x"] = torch.randn(num_nodes, input_dim_nodes)
    graph.edata["x"] = torch.randn(graph.num_edges(), input_dim_edges)

    offsets, indices, edge_perm = graph.adj_tensors("csc")

    partition0 = partition_graph_nodewise(
        offsets.to(dtype=torch.int64),
        indices.to(dtype=torch.int64),
        model_parallel_size,
        rank,
        device,
    )

    graph_multi_gpu = CuGraphCSC(
        offsets.to(device),
        indices.to(device),
        graph.num_src_nodes(),
        graph.num_dst_nodes(),
        partition_size=model_parallel_size,
        partition_group_name=pg_name,
        graph_partition=partition0,
    )

    dataset = BistrideMultiLayerGraphDataset(
        [graph_multi_gpu.to_dgl_graph()],
        num_layers,
    )

    pass


def test_partitioned_bsms():
    model_parallel_size = 1

    torch.multiprocessing.set_start_method("spawn", force=True)

    torch.multiprocessing.spawn(
        run_bsms_partitioning,
        args=(model_parallel_size,),
        nprocs=model_parallel_size,
    )
