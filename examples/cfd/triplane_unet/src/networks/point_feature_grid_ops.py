from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
# TODO(akamenev): migration
# import open3d as o3d
# import open3d.ml.torch as ml3d
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import functional as F

from .base_model import BaseModule
from .components.reductions import REDUCTION_TYPES
from .neighbor_ops import NeighborMLPConvLayer, NeighborRadiusSearchLayer
from .net_utils import MLP, PositionalEncoding
from .point_feature_conv import PointFeatureCat, PointFeatureConv, PointFeatureTransform
from .point_feature_ops import (
    GridFeatures,
    GridFeaturesMemoryFormat,
    PointFeatures,
    grid_init,
)


class AABBGridFeatures(GridFeatures):
    def __init__(
        self,
        aabb_max: Tuple[float, float, float],
        aabb_min: Tuple[float, float, float],
        resolution: Union[Int[Tensor, "3"], List[int]],
        pos_encode_dim: int = 32,
    ):
        grid = grid_init(aabb_max, aabb_min, resolution)
        feat = PositionalEncoding(pos_encode_dim, data_range=aabb_max[0] - aabb_min[0])(grid)
        super().__init__(grid, feat.view(*resolution, -1))


class PointFeatureToGrid(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aabb_max: Tuple[float, float, float],
        aabb_min: Tuple[float, float, float],
        voxel_size: Optional[float] = None,
        resolution: Optional[Union[Int[Tensor, "3"], List[int]]] = None,
        use_rel_pos: bool = True,
        use_rel_pos_encode: bool = False,
        pos_encode_dim: int = 32,
        reductions: List[REDUCTION_TYPES] = ["mean"],
        neighbor_search_type: Literal["radius", "knn"] = "radius",
        knn_k: int = 16,
    ) -> None:
        super().__init__()
        if resolution is None:
            assert voxel_size is not None
            resolution = (
                int((aabb_max[0] - aabb_min[0]) / voxel_size),
                int((aabb_max[1] - aabb_min[1]) / voxel_size),
                int((aabb_max[2] - aabb_min[2]) / voxel_size),
            )
        if voxel_size is None:
            assert resolution is not None
        if isinstance(resolution, Tensor):
            resolution = resolution.tolist()
        self.resolution = resolution
        for i in range(3):
            assert aabb_max[i] > aabb_min[i]
        self.grid_features = AABBGridFeatures(
            aabb_max, aabb_min, resolution, pos_encode_dim=pos_encode_dim
        )
        # Find per axis scaler that scales the vertices to [0, resolution[0]] x [0, resolution[1]] x [0, resolution[2]]
        vertices_scaler = torch.FloatTensor(
            [
                resolution[0] / (aabb_max[0] - aabb_min[0]),
                resolution[1] / (aabb_max[1] - aabb_min[1]),
                resolution[2] / (aabb_max[2] - aabb_min[2]),
            ]
        )
        self.conv = PointFeatureConv(
            radius=np.sqrt(3),  # diagonal of a unit cube
            in_channels=in_channels,
            out_channels=out_channels,
            provided_in_channels=3 * pos_encode_dim,
            use_rel_pos=use_rel_pos,
            use_rel_pos_encode=use_rel_pos_encode,
            pos_encode_dim=pos_encode_dim,
            neighbor_search_vertices_scaler=vertices_scaler,
            out_point_feature_type="provided",
            reductions=reductions,
            neighbor_search_type=neighbor_search_type,
            knn_k=knn_k,
        )

    def forward(self, point_features: PointFeatures) -> GridFeatures:
        # to device
        self.grid_features.to(device=point_features.vertices.device)
        out_point_features = self.conv(
            point_features,
            self.grid_features.point_features,
        )
        out_grid = out_point_features.to_grid_features(self.resolution)
        return out_grid


class GridFeatureToPoint(nn.Module):
    def __init__(
        self,
        grid_in_channels: int,
        point_in_channels: int,
        out_channels: int,
        aabb_max: Tuple[float, float, float],
        aabb_min: Tuple[float, float, float],
        use_rel_pos: bool = True,
        use_rel_pos_embed: bool = False,
        pos_embed_dim: int = 32,
        sample_method: Literal["graphconv", "interp"] = "graphconv",
        neighbor_search_type: Literal["radius", "knn"] = "radius",
        knn_k: int = 16,
        reductions: List[REDUCTION_TYPES] = ["mean"],
    ) -> None:
        super().__init__()
        self.sample_method = sample_method
        if sample_method == "graphconv":
            self.conv = GridFeatureToPointGraphConv(
                grid_in_channels,
                point_in_channels,
                out_channels,
                aabb_max,
                aabb_min,
                use_rel_pos=use_rel_pos,
                use_rel_pos_embed=use_rel_pos_embed,
                pos_embed_dim=pos_embed_dim,
                neighbor_search_type=neighbor_search_type,
                knn_k=knn_k,
                reductions=reductions,
            )
        elif sample_method == "interp":
            self.conv = GridFeatureToPointInterp(
                aabb_max,
                aabb_min,
                cat_in_point_features=True,
            )
            self.transform = PointFeatureTransform(
                nn.Sequential(
                    nn.Linear(grid_in_channels + point_in_channels, out_channels),
                    nn.LayerNorm(out_channels),
                )
            )
        else:
            raise NotImplementedError

    def forward(self, grid_features: GridFeatures, point_features: PointFeatures) -> PointFeatures:
        out_point_features = self.conv(grid_features, point_features)
        if self.sample_method == "interp":
            out_point_features = self.transform(out_point_features)
        return out_point_features


class GridFeatureToPointGraphConv(nn.Module):
    def __init__(
        self,
        grid_in_channels: int,
        point_in_channels: int,
        out_channels: int,
        aabb_max: Tuple[float, float, float],
        aabb_min: Tuple[float, float, float],
        use_rel_pos: bool = True,
        use_rel_pos_embed: bool = False,
        pos_embed_dim: int = 32,
        neighbor_search_type: Literal["radius", "knn"] = "radius",
        knn_k: int = 16,
        reductions: List[REDUCTION_TYPES] = ["mean"],
    ) -> None:
        super().__init__()
        self.aabb_max = aabb_max
        self.aabb_min = aabb_min
        self.conv = PointFeatureConv(
            radius=np.sqrt(3),  # diagonal of a unit cube
            in_channels=grid_in_channels,
            out_channels=out_channels,
            provided_in_channels=point_in_channels,
            use_rel_pos=use_rel_pos,
            use_rel_pos_encode=use_rel_pos_embed,
            pos_encode_dim=pos_embed_dim,
            out_point_feature_type="provided",
            neighbor_search_type=neighbor_search_type,
            knn_k=knn_k,
            reductions=reductions,
        )

    def forward(self, grid_features: GridFeatures, point_features: PointFeatures) -> PointFeatures:
        resolution = grid_features.resolution
        # Find per axis scaler that scales the vertices to [0, resolution[0]] x [0, resolution[1]] x [0, resolution[2]]
        vertices_scaler = torch.FloatTensor(
            [
                resolution[0] / (self.aabb_max[0] - self.aabb_min[0]),
                resolution[1] / (self.aabb_max[1] - self.aabb_min[1]),
                resolution[2] / (self.aabb_max[2] - self.aabb_min[2]),
            ]
        )
        out_point_features = self.conv(
            grid_features.point_features,
            point_features,
            neighbor_search_vertices_scaler=vertices_scaler,
        )
        return out_point_features


class GridFeatureToPointInterp(nn.Module):
    def __init__(
        self,
        aabb_max: Tuple[float, float, float],
        aabb_min: Tuple[float, float, float],
        cat_in_point_features: bool = True,
    ) -> None:
        super().__init__()
        self.aabb_max = torch.Tensor(aabb_max)
        self.aabb_min = torch.Tensor(aabb_min)
        self.cat_in_point_features = cat_in_point_features
        self.cat = PointFeatureCat()

    def to(self, *args, **kwargs):
        self.aabb_max = self.aabb_max.to(*args, **kwargs)
        self.aabb_min = self.aabb_min.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, grid_features: GridFeatures, point_features: PointFeatures) -> PointFeatures:
        # Use F.interpolate to interpolate grid features to point features
        grid_features.to(memory_format=GridFeaturesMemoryFormat.c_x_y_z)
        xyz = point_features.vertices  # N x 3
        self.to(device=xyz.device)
        normalized_xyz = (xyz - self.aabb_min) / (self.aabb_max - self.aabb_min) * 2 - 1
        normalized_xyz = normalized_xyz.view(1, 1, 1, -1, 3)
        batch_grid_features = grid_features.batch_features  # B x C x X x Y x Z
        # interpolate
        batch_point_features = (
            F.grid_sample(
                batch_grid_features,
                normalized_xyz,
                align_corners=True,
            )
            .squeeze()
            .permute(1, 0)
        )  # N x C

        out_point_features = PointFeatures(
            point_features.vertices,
            batch_point_features,
        )
        if self.cat_in_point_features:
            out_point_features = self.cat(point_features, out_point_features)
        return out_point_features


class PointFeatureToDistanceGridFeature(BaseModule):
    def __init__(
        self,
        aabb_max: Tuple[float, float, float],
        aabb_min: Tuple[float, float, float],
        grid_resolution: Optional[Tuple[int, int, int]] = None,
        voxel_size: Optional[float] = None,
        pos_encode_dist: bool = True,
        pos_encode_grid: bool = True,
        pos_encode_dim: int = 32,
    ):
        super().__init__()
        self.aabb_max = aabb_max
        self.aabb_min = aabb_min
        self.pos_encoding = PositionalEncoding(
            pos_encode_dim,
            data_range=aabb_max[0] - aabb_min[0],
        )
        self.pos_encode_dist = pos_encode_dist
        self.pos_encode_grid = pos_encode_grid
        self.pos_encode_dim = pos_encode_dim
        if voxel_size is not None:
            assert grid_resolution is None
        if grid_resolution is None:
            assert voxel_size is not None
            grid_resolution = (
                int((aabb_max[0] - aabb_min[0]) / voxel_size),
                int((aabb_max[1] - aabb_min[1]) / voxel_size),
                int((aabb_max[2] - aabb_min[2]) / voxel_size),
            )
        print(f"Using grid resolution: {grid_resolution}")
        self.grid_resolution = grid_resolution
        self.grid_features = AABBGridFeatures(
            aabb_max, aabb_min, grid_resolution, pos_encode_dim=pos_encode_dim
        )

    @property
    def num_channels(self):
        return (self.pos_encode_dim if self.pos_encode_dist else 1) + int(
            self.pos_encode_grid
        ) * 3 * self.pos_encode_dim

    def forward(self, point_features: PointFeatures) -> GridFeatures:
        # Use open3d to find distance
        vertices = point_features.vertices.cpu()
        query_pts = self.grid_features.vertices.reshape(-1, 3).cpu()
        k = 1
        knn_result = ml3d.ops.knn_search(
            vertices,
            query_pts,
            k=k,
            points_row_splits=torch.LongTensor([0, len(vertices)]),
            queries_row_splits=torch.LongTensor([0, len(query_pts)]),
            return_distances=True,
        )
        distances = knn_result.neighbors_distance.reshape(*self.grid_resolution, 1)
        if self.pos_encode_dist:
            distances = self.pos_encoding(distances)
        if self.pos_encode_grid:
            distances = torch.cat(
                [distances, self.grid_features.features],
                dim=-1,
            )
        grid_features = GridFeatures(
            self.grid_features.vertices,
            distances,
        ).to(device=point_features.vertices.device)
        return grid_features


class GridFeatureToPointFeature(BaseModule):
    def __init__(
        self,
        in_channels: int,
        pos_encode_point: bool = True,
        pos_encode_range: float = 1.0,
        pos_encode_dim: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.encoder = PositionalEncoding(pos_encode_dim, data_range=pos_encode_range)
        self.pos_encode_point = pos_encode_point
        self.pos_encode_dim = pos_encode_dim

    @property
    def num_channels(self):
        return self.in_channels + (self.pos_encode_dim * 3 if self.pos_encode_point else 0)

    def forward(self, grid_features: GridFeatures, point_features: PointFeatures) -> PointFeatures:
        # Use F.interpolate to interpolate grid features to point features
        assert grid_features.memory_format == GridFeaturesMemoryFormat.c_x_y_z
        batch_grid_features = grid_features.batch_features  # B x C x X x Y x Z
        # normalize the point_features.vertices to [-1, 1] using grid_features.vertices min max
        vertices = point_features.vertices
        grid_min = grid_features.vertices[0, 0, 0]
        grid_max = grid_features.vertices[-1, -1, -1]
        vertices = (vertices - grid_min) / (grid_max - grid_min) * 2 - 1
        # interpolate
        batch_point_features = (
            torch.nn.functional.grid_sample(
                batch_grid_features,
                vertices.view(1, 1, 1, -1, 3),
                align_corners=True,
            )
            .squeeze()
            .permute(1, 0)
        )
        if self.pos_encode_point:
            vert_pos = self.encoder(point_features.vertices)
            # concat vert_pos to batch_point_features
            batch_point_features = torch.cat([batch_point_features, vert_pos], dim=-1)
        point_features = PointFeatures(
            point_features.vertices,
            batch_point_features,
        )
        return point_features


class GridFeatureCat(BaseModule):
    def forward(
        self, grid_features: GridFeatures, other_grid_features: GridFeatures
    ) -> GridFeatures:
        assert grid_features.memory_format == other_grid_features.memory_format
        # assert torch.allclose(grid_features.vertices, other_grid_features.vertices)

        orig_memory_format = grid_features.memory_format
        grid_features.to(memory_format=GridFeaturesMemoryFormat.c_x_y_z)
        other_grid_features.to(memory_format=GridFeaturesMemoryFormat.c_x_y_z)
        cat_grid_features = GridFeatures(
            vertices=grid_features.vertices,
            features=torch.cat([grid_features.features, other_grid_features.features], dim=0),
            memory_format=grid_features.memory_format,
            grid_shape=grid_features.grid_shape,
            num_channels=grid_features.num_channels + other_grid_features.num_channels,
        )
        cat_grid_features.to(memory_format=orig_memory_format)
        return cat_grid_features


class GridFeatureFNO(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resolution: Tuple[int, int, int] = (32, 32, 32),
        hidden_channels: int = 86,
        domain_padding: float = 0.0,
        norm: Literal["group_norm"] = "group_norm",
        factorization: Literal["tucker"] = "tucker",
        rank: float = 0.4,
    ):
        from neuralop.models import FNO

        super().__init__()
        self.fno = FNO(
            resolution,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            use_mlp=True,
            mlp={"expansion": 1.0, "dropout": 0},
            domain_padding=domain_padding,
            factorization=factorization,
            norm=norm,
            rank=rank,
        )

    def forward(self, grid_features: GridFeatures) -> GridFeatures:
        grid_features = grid_features.to(memory_format=GridFeaturesMemoryFormat.c_x_y_z)
        batch_features = self.fno(grid_features.batch_features)
        out_grid_features = GridFeatures(
            grid_features.vertices,
            batch_features.squeeze(0),
            memory_format=grid_features.memory_format,
            grid_shape=grid_features.grid_shape,
            num_channels=self.fno.out_channels,
        )
        return out_grid_features