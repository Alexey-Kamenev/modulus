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
# ruff: noqa: E402

import numpy as np
import pytest
import torch
from pytest_utils import import_or_fail
from torch.testing import assert_close


@import_or_fail(["dgl", "torch_geometric"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("aggregation", ["sum", "mean"])
def test_mesh_node_block_dgl_pyg_equivalence(device, aggregation, pytestconfig):
    """Test that MeshNodeBlock produces equivalent outputs for DGL and PyG graphs."""
    # (DGL2PYG): remove this once DGL is removed.

    import dgl
    from torch_geometric.data import Data as PyGData

    from physicsnemo.models.gnn_layers.mesh_node_block import MeshNodeBlock

    # Set seeds for reproducibility.
    torch.manual_seed(42)
    dgl.seed(42)
    np.random.seed(42)

    # Test parameters.
    num_nodes = 10
    num_edges = 20
    input_dim_nodes = 8
    input_dim_edges = 8
    output_dim = 8
    hidden_dim = 10

    # Create MeshNodeBlock.
    node_block = MeshNodeBlock(
        aggregation=aggregation,
        input_dim_nodes=input_dim_nodes,
        input_dim_edges=input_dim_edges,
        output_dim=input_dim_edges,
        hidden_dim=hidden_dim,
        hidden_layers=1,
    ).to(device)

    # Create random edge connectivity.
    src_nodes = torch.randint(0, num_nodes, (num_edges,))
    dst_nodes = torch.randint(0, num_nodes, (num_edges,))

    # Create node and edge features.
    node_features = torch.randn(num_nodes, input_dim_nodes, device=device)
    edge_features = torch.randn(num_edges, input_dim_edges, device=device)

    # Create DGL graph.
    dgl_graph = dgl.graph((src_nodes, dst_nodes)).to(device)

    # Create PyG graph.
    edge_index = torch.stack([src_nodes, dst_nodes], dim=0).to(device)
    pyg_graph = PyGData(edge_index=edge_index)

    # Forward pass with DGL graph.
    efeat_dgl, nfeat_dgl = node_block(
        efeat=edge_features,
        nfeat=node_features,
        graph=dgl_graph,
    )

    # Forward pass with PyG graph.
    efeat_pyg, nfeat_pyg = node_block(
        efeat=edge_features,
        nfeat=node_features,
        graph=pyg_graph,
    )

    # Verify outputs are equivalent.
    assert_close(efeat_dgl, efeat_pyg)
    assert_close(
        nfeat_dgl,
        nfeat_pyg,
    )

    # Verify output shapes.
    assert efeat_dgl.shape == (num_edges, input_dim_edges)
    assert nfeat_dgl.shape == (num_nodes, output_dim)
    assert efeat_pyg.shape == (num_edges, input_dim_edges)
    assert nfeat_pyg.shape == (num_nodes, output_dim)


@import_or_fail(["dgl", "torch_geometric"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_mesh_node_block_gradient_equivalence(device, pytestconfig):
    """Test that MeshNodeBlock produces equivalent gradients for DGL and PyG graphs."""
    # (DGL2PYG): remove this once DGL is removed.

    import dgl
    from torch_geometric.data import Data as PyGData

    from physicsnemo.models.gnn_layers.mesh_node_block import MeshNodeBlock

    # Set seeds for reproducibility.
    torch.manual_seed(123)
    dgl.seed(123)
    np.random.seed(123)

    # Test parameters.
    num_nodes = 8
    num_edges = 10
    input_dim_nodes = 4
    input_dim_edges = 4
    output_dim = 4

    # Create identical MeshNodeBlocks.
    node_block_dgl = MeshNodeBlock(
        aggregation="sum",
        input_dim_nodes=input_dim_nodes,
        input_dim_edges=input_dim_edges,
        output_dim=output_dim,
        hidden_dim=8,
        hidden_layers=1,
    ).to(device)

    node_block_pyg = MeshNodeBlock(
        aggregation="sum",
        input_dim_nodes=input_dim_nodes,
        input_dim_edges=input_dim_edges,
        output_dim=output_dim,
        hidden_dim=8,
        hidden_layers=1,
    ).to(device)

    # Copy weights to ensure identical models.
    node_block_pyg.load_state_dict(node_block_dgl.state_dict())

    # Create random edge connectivity.
    src_nodes = torch.randint(0, num_nodes, (num_edges,))
    dst_nodes = torch.randint(0, num_nodes, (num_edges,))

    # Create node and edge features (requires_grad for gradient test).
    node_features_dgl = torch.randn(
        num_nodes, input_dim_nodes, device=device, requires_grad=True
    )
    edge_features_dgl = torch.randn(
        num_edges, input_dim_edges, device=device, requires_grad=True
    )

    node_features_pyg = node_features_dgl.clone().detach().requires_grad_(True)
    edge_features_pyg = edge_features_dgl.clone().detach().requires_grad_(True)

    # Create DGL graph.
    dgl_graph = dgl.graph((src_nodes, dst_nodes)).to(device)

    # Create PyG graph.
    edge_index = torch.stack([src_nodes, dst_nodes], dim=0).to(device)
    pyg_graph = PyGData(edge_index=edge_index)

    # Forward pass with DGL graph.
    efeat_dgl, nfeat_dgl = node_block_dgl(
        efeat=edge_features_dgl,
        nfeat=node_features_dgl,
        graph=dgl_graph,
    )

    # Forward pass with PyG graph.
    efeat_pyg, nfeat_pyg = node_block_pyg(
        efeat=edge_features_pyg,
        nfeat=node_features_pyg,
        graph=pyg_graph,
    )

    # Create identical loss functions.
    loss_dgl = nfeat_dgl.sum()
    loss_pyg = nfeat_pyg.sum()

    # Backward pass.
    loss_dgl.backward()
    loss_pyg.backward()

    # Compare gradients.
    assert_close(
        node_features_dgl.grad,
        node_features_pyg.grad,
    )
    assert_close(
        edge_features_dgl.grad,
        edge_features_pyg.grad,
    )

    # Compare model parameter gradients.
    for (name_dgl, param_dgl), (name_pyg, param_pyg) in zip(
        node_block_dgl.named_parameters(), node_block_pyg.named_parameters()
    ):
        assert (
            name_dgl == name_pyg
        ), f"Parameter names should match: {name_dgl} vs {name_pyg}"
        assert_close(
            param_dgl.grad,
            param_pyg.grad,
        )


@import_or_fail(["dgl", "torch_geometric"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_mesh_node_block_batched_equivalence(device, pytestconfig):
    """Test that MeshNodeBlock produces equivalent outputs for batched DGL and PyG graphs."""
    # (DGL2PYG): remove this once DGL is removed.

    import dgl
    from torch_geometric.data import Batch
    from torch_geometric.data import Data as PyGData

    from physicsnemo.models.gnn_layers.mesh_node_block import MeshNodeBlock

    # Set seeds for reproducibility.
    torch.manual_seed(456)
    dgl.seed(456)
    np.random.seed(456)

    # Test parameters.
    batch_size = 3
    num_nodes_per_graph = 6
    num_edges_per_graph = 10
    input_dim_nodes = 4
    input_dim_edges = 4
    output_dim = 4

    # Create MeshNodeBlock.
    node_block = MeshNodeBlock(
        aggregation="mean",
        input_dim_nodes=input_dim_nodes,
        input_dim_edges=input_dim_edges,
        output_dim=output_dim,
        hidden_dim=10,
        hidden_layers=1,
    ).to(device)

    # Create batch of DGL graphs.
    dgl_graphs = []
    pyg_graphs = []

    for _ in range(batch_size):
        # Create random edge connectivity.
        src_nodes = torch.randint(0, num_nodes_per_graph, (num_edges_per_graph,))
        dst_nodes = torch.randint(0, num_nodes_per_graph, (num_edges_per_graph,))

        # Create DGL graph.
        dgl_graph = dgl.graph((src_nodes, dst_nodes))
        dgl_graphs.append(dgl_graph)

        # Create PyG graph.
        edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
        pyg_graph = PyGData(edge_index=edge_index, num_nodes=num_nodes_per_graph)
        pyg_graphs.append(pyg_graph)

    # Batch graphs.
    batched_dgl_graph = dgl.batch(dgl_graphs).to(device)
    batched_pyg_graph = Batch.from_data_list(pyg_graphs).to(device)

    # Create batched features.
    total_nodes = batch_size * num_nodes_per_graph
    total_edges = batch_size * num_edges_per_graph

    node_features = torch.randn(total_nodes, input_dim_nodes, device=device)
    edge_features = torch.randn(total_edges, input_dim_edges, device=device)

    # Forward pass with batched DGL graph.
    efeat_dgl, nfeat_dgl = node_block(
        efeat=edge_features,
        nfeat=node_features,
        graph=batched_dgl_graph,
    )

    # Forward pass with batched PyG graph.
    efeat_pyg, nfeat_pyg = node_block(
        efeat=edge_features,
        nfeat=node_features,
        graph=batched_pyg_graph,
    )

    # Verify outputs are equivalent.
    assert_close(efeat_dgl, efeat_pyg)
    assert_close(nfeat_dgl, nfeat_pyg)

    # Verify output shapes.
    assert efeat_dgl.shape == (total_edges, input_dim_edges)
    assert nfeat_dgl.shape == (total_nodes, output_dim)
    assert efeat_pyg.shape == (total_edges, input_dim_edges)
    assert nfeat_pyg.shape == (total_nodes, output_dim)
