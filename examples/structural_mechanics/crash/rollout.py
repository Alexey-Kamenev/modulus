# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as ckpt
from typing import List

from physicsnemo.models.transolver import Transolver
from physicsnemo.models.meshgraphnet import MeshGraphNet
from physicsnemo.models.figconvnet.figconvunet import FIGConvUNet

from datapipe import SimSample

EPS = 1e-8


class TransolverAutoregressiveRolloutTraining(Transolver):
    """
    Transolver model with autoregressive rollout training.

    Predicts sequence by autoregressively updating velocity and position
    using predicted accelerations. Supports gradient checkpointing during training.
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        """
        Args:
            sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        node_features = sample.node_features  # [N,F_in]
        N = sample.node_features.size(0)
        device = sample.node_features.device

        # Initial states
        y_t1 = node_features[..., :3]  # [N,3]
        thickness = node_features[..., -1:]  # [N,1]
        y_t0 = y_t1 - self.initial_vel * self.dt  # backstep using initial velocity

        outputs: List[torch.Tensor] = []
        for t in range(self.rollout_steps):
            time_t = 0.0 if self.rollout_steps <= 1 else t / (self.rollout_steps - 1)
            time_t = torch.tensor([time_t], device=device, dtype=torch.float32)

            # Velocity normalization
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )

            # Model input
            fx_t = torch.cat(
                [vel_norm, thickness, time_t.expand(N, 1)], dim=-1
            )  # [N, 3+1+1]

            def step_fn(fx, embedding):
                return super(TransolverAutoregressiveRolloutTraining, self).forward(
                    fx=fx, embedding=embedding
                )

            if self.training:
                outf = ckpt(
                    step_fn, fx_t.unsqueeze(0), y_t1.unsqueeze(0), use_reentrant=False
                ).squeeze(0)
            else:
                outf = step_fn(fx_t.unsqueeze(0), y_t1.unsqueeze(0)).squeeze(0)

            # De-normalize acceleration
            acc = (
                outf * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )
            vel = self.dt * acc + vel
            y_t2 = self.dt * vel + y_t1

            outputs.append(y_t2)
            y_t1, y_t0 = y_t2, y_t1

        return torch.stack(outputs, dim=0)  # [T,N,3]


class TransolverTimeConditionalRollout(Transolver):
    """
    Transolver model with time-conditional rollout.

    Predicts each time step independently, conditioned on normalized time.
    """

    def __init__(self, *args, **kwargs):
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(
        self,
        sample: SimSample,
        data_stats: dict,
    ) -> torch.Tensor:
        """
        Args:
            Sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        node_features = sample.node_features  # [N,4] (pos(3) + thickness(1))
        assert node_features.ndim == 2 and node_features.shape[1] == 4, (
            f"Expected node_features [N,4], got {node_features.shape}"
        )

        x = node_features[..., :3]  # initial pos
        thickness = node_features[..., -1:]
        outputs: List[torch.Tensor] = []
        time_seq = torch.linspace(0.0, 1.0, self.rollout_steps, device=x.device)

        for time in time_seq:
            fx_t = thickness  # [N,1]

            def step_fn(fx, embedding, time_t):
                return super(TransolverTimeConditionalRollout, self).forward(
                    fx=fx, embedding=embedding, time=time_t
                )

            if self.training:
                outf = ckpt(
                    step_fn,
                    fx_t.unsqueeze(0),
                    x.unsqueeze(0),
                    time.unsqueeze(0),
                    use_reentrant=False,
                ).squeeze(0)
            else:
                outf = step_fn(
                    fx_t.unsqueeze(0), x.unsqueeze(0), time.unsqueeze(0)
                ).squeeze(0)

            y_t2 = x + outf
            outputs.append(y_t2)

        return torch.stack(outputs, dim=0)  # [T,N,3]


class MeshGraphNetAutoregressiveRolloutTraining(MeshGraphNet):
    """MeshGraphNet with autoregressive rollout training."""

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        """
        Args:
            Sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        node_features = sample.node_features
        edge_features = sample.graph.edge_attr
        graph = sample.graph

        N = node_features.size(0)
        y_t1 = node_features[..., :3]
        thickness = node_features[..., -1:]
        y_t0 = y_t1 - self.initial_vel * self.dt

        outputs: List[torch.Tensor] = []
        for _ in range(self.rollout_steps):
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )
            fx_t = torch.cat([y_t1, vel_norm, thickness], dim=-1)

            def step_fn(nf, ef, g):
                return super(MeshGraphNetAutoregressiveRolloutTraining, self).forward(
                    node_features=nf, edge_features=ef, graph=g
                )

            outf = (
                ckpt(step_fn, fx_t, edge_features, graph, use_reentrant=False)
                if self.training
                else step_fn(fx_t, edge_features, graph)
            )

            acc = (
                outf * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )

            vel = self.dt * acc + vel
            y_t2 = self.dt * vel + y_t1

            outputs.append(y_t2)
            y_t1, y_t0 = y_t2, y_t1

        return torch.stack(outputs, dim=0)


class MeshGraphNetTimeConditionalRollout(MeshGraphNet):
    """MeshGraphNet with time-conditional rollout."""

    def __init__(self, *args, **kwargs):
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        """
        Args:
            Sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        node_features = sample.node_features
        edge_features = sample.graph.edge_attr
        graph = sample.graph

        x = node_features[..., :3]
        thickness = node_features[..., -1:]
        outputs: List[torch.Tensor] = []
        time_seq = torch.linspace(0.0, 1.0, self.rollout_steps, device=x.device)

        for time in time_seq:
            fx_t = torch.cat([x, thickness, time.expand(x.size(0), 1)], dim=-1)

            def step_fn(nf, ef, g):
                return super(MeshGraphNetTimeConditionalRollout, self).forward(
                    node_features=nf, edge_features=ef, graph=g
                )

            outf = (
                ckpt(step_fn, fx_t, edge_features, graph, use_reentrant=False)
                if self.training
                else step_fn(fx_t, edge_features, graph)
            )

            y_t2 = x + outf
            outputs.append(y_t2)

        return torch.stack(outputs, dim=0)


class TransolverOneStepRollout(
    Transolver
):  # TODO this can be merged with TransolverAutoregressiveRolloutTraining
    """
    One-step rollout:
      - Training: teacher forcing (uses GT for each step, but first step needs backstep)
      - Inference: autoregressive (uses predictions)
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt", 5e-3)
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        N = sample.node_features.size(0)
        thickness = sample.node_features[..., -1:]  # [N,1]

        # Ground truth sequence [T,N,3]
        gt_seq = torch.cat(
            [
                sample.node_features[..., :3].unsqueeze(0),  # pos_t0
                sample.node_target.view(N, -1, 3).transpose(0, 1),  # pos_t1..pos_T
            ],
            dim=0,
        )

        outputs: List[torch.Tensor] = []

        # First step: backstep to create y_-1
        y_t0 = gt_seq[0] - self.initial_vel * self.dt
        y_t1 = gt_seq[0]

        for t in range(self.rollout_steps):
            if self.training and t > 0:
                # teacher forcing uses GT pairs
                y_t0, y_t1 = gt_seq[t - 1], gt_seq[t]

            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )
            fx_t = torch.cat([vel_norm, thickness], dim=-1)

            def step_fn(fx, embedding):
                return super(TransolverOneStepRollout, self).forward(
                    fx=fx, embedding=embedding
                )

            if self.training:
                outf = ckpt(
                    step_fn, fx_t.unsqueeze(0), y_t1.unsqueeze(0), use_reentrant=False
                ).squeeze(0)
            else:
                outf = step_fn(fx_t.unsqueeze(0), y_t1.unsqueeze(0)).squeeze(0)

            acc = (
                outf * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )
            vel_pred = self.dt * acc + vel
            y_t2_pred = self.dt * vel_pred + y_t1

            outputs.append(y_t2_pred)

            if not self.training:
                # autoregressive update for inference
                y_t0, y_t1 = y_t1, y_t2_pred

        return torch.stack(outputs, dim=0)  # [T,N,3]


class MeshGraphNetOneStepRollout(MeshGraphNet):
    """
    MeshGraphNet with one-step rollout:
      - Training: teacher forcing (uses GT positions at each step, first step needs backstep)
      - Inference: autoregressive (uses predictions)
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt", 5e-3)
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        node_features = sample.node_features
        edge_features = sample.graph.edge_attr
        graph = sample.graph

        N = node_features.size(0)
        thickness = node_features[..., -1:]

        # Full ground truth trajectory [T,N,3]
        gt_seq = torch.cat(
            [
                node_features[..., :3].unsqueeze(0),  # pos_t0
                sample.node_target.view(N, -1, 3).transpose(0, 1),  # pos_t1..T
            ],
            dim=0,
        )

        outputs: List[torch.Tensor] = []

        # First step: construct backstep
        y_t0 = gt_seq[0] - self.initial_vel * self.dt
        y_t1 = gt_seq[0]

        for t in range(self.rollout_steps):
            if self.training and t > 0:
                # Teacher forcing: use GT sequence
                y_t0, y_t1 = gt_seq[t - 1], gt_seq[t]

            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )

            fx_t = torch.cat([y_t1, vel_norm, thickness], dim=-1)

            def step_fn(nf, ef, g):
                return super(MeshGraphNetOneStepRollout, self).forward(
                    node_features=nf, edge_features=ef, graph=g
                )

            if self.training:
                outf = ckpt(step_fn, fx_t, edge_features, graph, use_reentrant=False)
            else:
                outf = step_fn(fx_t, edge_features, graph)

            acc = (
                outf * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )
            vel_pred = self.dt * acc + vel
            y_t2_pred = self.dt * vel_pred + y_t1

            outputs.append(y_t2_pred)

            if not self.training:
                # Autoregressive update
                y_t0, y_t1 = y_t1, y_t2_pred

        return torch.stack(outputs, dim=0)  # [T,N,3]


class FIGConvUNetTimeConditionalRollout(FIGConvUNet):
    """
    FIGConvUNet with time-conditional rollout for crash simulation.

    Predicts each time step independently, conditioned on normalized time.
    """

    def __init__(self, *args, **kwargs):
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(
        self,
        sample: SimSample,
        data_stats: dict,
    ) -> torch.Tensor:
        """
        Args:
            Sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [T, N, 3] rollout of predicted positions
        """
        node_features = sample.node_features  # [N, 4] (pos(3) + thickness(1))
        assert node_features.ndim == 2 and node_features.shape[1] == 4, (
            f"Expected node_features [N,4], got {node_features.shape}"
        )

        x = node_features[..., :3]  # initial pos [N, 3]
        thickness = node_features[..., -1:]  # [N, 1]

        outputs: List[torch.Tensor] = []
        time_seq = torch.linspace(0.0, 1.0, self.rollout_steps, device=x.device)

        for time_t in time_seq:
            # Prepare vertices for FIGConvUNet: [1, N, 3]
            vertices = x.unsqueeze(0)  # [1, N, 3]

            # Prepare features: thickness + time [N, 2]
            time_expanded = time_t.expand(x.size(0), 1)  # [N, 1]
            features = torch.cat([thickness, time_expanded], dim=-1)  # [N, 2]
            features = features.unsqueeze(0)  # [1, N, 2]

            def step_fn(verts, feats):
                out, _ = super(FIGConvUNetTimeConditionalRollout, self).forward(
                    vertices=verts, features=feats
                )
                return out

            if self.training:
                outf = ckpt(
                    step_fn,
                    vertices,
                    features,
                    use_reentrant=False,
                ).squeeze(0)  # [N, 3]
            else:
                outf = step_fn(vertices, features).squeeze(0)  # [N, 3]

            y_t = x + outf
            outputs.append(y_t)

        return torch.stack(outputs, dim=0)  # [T, N, 3]
