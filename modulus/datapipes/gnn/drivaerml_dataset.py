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

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import dgl
import pandas as pd
import torch
import yaml
from dgl.data import DGLDataset
from torch import Tensor

from modulus.datapipes.datapipe import Datapipe
from modulus.datapipes.meta import DatapipeMetaData

try:
    import pyvista as pv
    import vtk
except ImportError:
    raise ImportError(
        "DrivAerML Dataset requires the vtk and pyvista libraries. "
        "Install with pip install vtk pyvista"
    )


POS_KEY: str = "pos"
P_KEY: str = "pMeanTrim"
WSS_KEY: str = "wallShearStressMeanTrim"


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "DrivAerML"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = False
    # Parallel
    ddp_sharding: bool = True


class DrivAerMLDataset(DGLDataset, Datapipe):
    """
    DrivAerML dataset.

    Note: DrivAerMLDataset does not use default DGLDataset caching
    functionality such as `has_cache`, `download` etc,
    as it is invoked during the __init__ call so takes a lot of time.
    Instead, DrivAerMLDataset caches graphs in __getitem__ call thus
    avoiding long initialization delay.

    Parameters
    ----------
    data_dir: str
        The directory where the data is stored.
    split: str, optional
        The dataset split. Can be 'train', 'validation', or 'test', by default 'train'.
    num_samples: int, optional
        The number of samples to use, by default 10.
    invar_keys: Iterable[str], optional
        The input node features to consider. Default includes 'pos'.
    outvar_keys: Iterable[str], optional
        The output features to consider. Default includes 'pMeanTrim' and 'wallShearStressMeanTrim'.
    normalize_keys: Iterable[str], optional
        The features to normalize. Default includes 'pMeanTrim' and 'wallShearStressMeanTrim'.
    decimate_pct: int, optional.
        Which decimation ratio to use, default is 5%. Currently can be one of [5, 10, 20].
    topology_preserving: bool.
        Whether to use topology-preserving variant of data.
    cache_dir: str, optional
        Path to the cache directory to store graphs in DGL format for fast loading.
        Default is ./dgl_cache/.
    force_reload: bool, optional
        If True, forces a reload of the data, by default False.
    name: str, optional
        The name of the dataset, by default 'dataset'.
    verbose: bool, optional
        If True, enables verbose mode, by default False.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        num_samples: int = 10,
        invar_keys: Iterable[str] = (POS_KEY,),
        outvar_keys: Iterable[str] = (P_KEY, WSS_KEY),
        normalize_keys: Iterable[str] = (P_KEY, WSS_KEY),
        decimate_pct: int = 5,
        topology_preserving: bool = False,
        cache_dir: str | Path = "./dgl_cache/",
        force_reload: bool = False,
        name: str = "dataset",
        verbose: bool = False,
        **kwargs,
    ) -> None:
        DGLDataset.__init__(self, name=name, force_reload=force_reload, verbose=verbose)
        Datapipe.__init__(self, meta=MetaData())

        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise ValueError(
                f"Path {self.data_dir} does not exist or is not a directory."
            )

        self.split = split.lower()
        if split not in (splits := ["train", "val", "test"]):
            raise ValueError(f"{split = } is not supported, must be one of {splits}.")

        self.run_dirs = pd.read_csv(
            self.data_dir / f"{self.split}_runs.txt", header=None, index_col=0
        )

        self.num_samples = num_samples
        self.input_keys = list(invar_keys)
        self.output_keys = list(outvar_keys)
        self.normalize_keys = list(normalize_keys)

        self.decimate_pct = decimate_pct
        self.topology_preserving = topology_preserving

        self.cache_dir: Path = (
            self._get_cache_dir(self.data_dir, Path(cache_dir))
            if cache_dir is not None
            else None
        )

        if self.num_samples > len(self.run_dirs):
            raise ValueError(
                f"Number of available {self.split} dataset entries "
                f"({len(self.run_dirs)}) is less than the number of samples "
                f"({self.num_samples})"
            )
        self.run_dirs = self.run_dirs.iloc[: self.num_samples]

        # TODO(akamenev): these are estimates from small sample, need to compute from full data.
        self.nstats = {
            k: {"mean": v[0], "std": v[1]}
            for k, v in {
                P_KEY: (-180.986, 227.377),
                WSS_KEY: (
                    torch.tensor([-0.9298492 , -0.00185992, -0.03872167]),
                    torch.tensor([1.5299146, 0.9425125, 1.1832976]),
                ),
            }.items()
        }

        self.estats = {
            "x": {
                "mean": torch.tensor([0, 0, 0, 0.011]),
                "std": torch.tensor([0.0124, 0.0086, 0.0074, 0.0130]),
            }
        }

    def __len__(self) -> int:
        return len(self.run_dirs)

    def __getitem__(self, idx: int) -> dgl.DGLGraph:
        if not 0 <= idx < len(self):
            raise IndexError(f"Invalid {idx = }, must be in [0, {len(self)})")

        run_dir = self.run_dirs.iloc[idx]
        run_name = run_dir.name

        if self.cache_dir is None:
            # Caching is disabled - create the graph.
            graph = self._create_dgl_graph(run_name)
        else:
            cached_graph_filename = self.cache_dir / (
                f"{run_name}_{self._get_vtp_filename()}.bin"
            )
            if not self._force_reload and cached_graph_filename.is_file():
                gs, _ = dgl.load_graphs(str(cached_graph_filename))
                if len(gs) != 1:
                    raise ValueError(f"Expected to load 1 graph but got {len(gs)}.")
                graph = gs[0]
            else:
                graph = self._create_dgl_graph(run_name)
                # self.cache_dir.mkdir(parents=True, exist_ok=True)
                dgl.save_graphs(str(cached_graph_filename), [graph])

        # Set graph inputs/outputs.
        graph.ndata["x"] = torch.cat([graph.ndata[k] for k in self.input_keys], dim=-1)
        graph.ndata["y"] = torch.cat([graph.ndata[k] for k in self.output_keys], dim=-1)

        return {
            # "name": run_name,
            "graph": graph,
        }

    @staticmethod
    def _get_cache_dir(data_dir, cache_dir):
        if not cache_dir.is_absolute():
            cache_dir = data_dir / cache_dir
        return cache_dir.resolve()

    def _get_vtp_filename(self):
        pro_str = "-pro" if self.topology_preserving else ""
        return f"boundary_decimate{pro_str}_{self.decimate_pct}_vars.vtp"

    def _create_dgl_graph(
        self,
        name: str,
        to_bidirected: bool = True,
        dtype: torch.dtype | str = torch.int32,
    ) -> dgl.DGLGraph:
        """Creates a DGL graph from DrivAerNet VTK data.

        Parameters
        ----------
        name : str
            Name of the graph in DrivAerNet.
        to_bidirected : bool, optional
            Whether to make the graph bidirected. Default is True.
        dtype : torch.dtype or str, optional
            Data type for the graph. Default is torch.int32.

        Returns
        -------
        dgl.DGLGraph
            The DGL graph.
        """

        def extract_edges(mesh: pv.PolyData) -> list[tuple[int, int]]:
            # Extract connectivity information from the mesh.
            # Traversal API is faster comparing to iterating over mesh.cell.
            polys = mesh.GetPolys()
            if polys is None:
                raise ValueError("Failed to get polygons from the mesh.")

            polys.InitTraversal()

            edge_list = []
            for _ in range(polys.GetNumberOfCells()):
                id_list = vtk.vtkIdList()
                polys.GetNextCell(id_list)
                num_ids = id_list.GetNumberOfIds()
                for j in range(num_ids - 1):
                    edge_list.append(  # noqa: PERF401
                        (id_list.GetId(j), id_list.GetId(j + 1))
                    )
                # Add the final edge between the last and the first vertices.
                edge_list.append((id_list.GetId(num_ids - 1), id_list.GetId(0)))

            return edge_list

        mesh = pv.read((self.data_dir / name) / self._get_vtp_filename())
        edge_list = extract_edges(mesh)

        # Create DGL graph using the connectivity information
        graph = dgl.graph(edge_list, idtype=dtype)
        if to_bidirected:
            graph = dgl.to_bidirected(graph)

        # Assign node features using the vertex data
        graph.ndata["pos"] = torch.tensor(mesh.points, dtype=torch.float32)

        point_data = mesh.cell_data_to_point_data()
        for k in (P_KEY, WSS_KEY):
            if k in self.output_keys:
                graph.ndata[k] = torch.tensor(point_data[k], dtype=torch.float32)

        # Normalize nodes.
        for k in self.input_keys + self.output_keys:
            if k not in self.normalize_keys:
                continue
            v = (graph.ndata[k] - self.nstats[k]["mean"]) / self.nstats[k]["std"]
            graph.ndata[k] = v.unsqueeze(-1) if v.ndim == 1 else v

        # Add edge features which contain relative edge nodes displacement and
        # displacement norm. Stored as `x` in the graph edge data.
        u, v = graph.edges()
        pos = graph.ndata["pos"]
        disp = pos[u] - pos[v]
        disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
        graph.edata["x"] = torch.cat((disp, disp_norm), dim=-1)

        # Normalize edges.
        for k, v in graph.edata.items():
            v = (v - self.estats[k]["mean"]) / self.estats[k]["std"]
            graph.edata[k] = v.unsqueeze(-1) if v.ndim == 1 else v

        return graph

    @torch.no_grad
    def denormalize(
        self, pred: Tensor, gt: Tensor, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Denormalizes the inputs using previously collected statistics."""

        def denorm(x: Tensor, name: str):
            stats = self.nstats[name]
            mean = torch.as_tensor(stats["mean"]).to(device)
            std = torch.as_tensor(stats["std"]).to(device)
            return x * std + mean

        pred_d = []
        gt_d = []
        pred_d.append(denorm(pred[:, :1], P_KEY))
        gt_d.append(denorm(gt[:, :1], P_KEY))

        if (k := WSS_KEY) in self.output_keys:
            pred_d.append(denorm(pred[:, 1:4], k))
            gt_d.append(denorm(gt[:, 1:4], k))

        return torch.cat(pred_d, dim=-1), torch.cat(gt_d, dim=-1)
