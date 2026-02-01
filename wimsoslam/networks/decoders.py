# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 ams-OSRAM AG
# Copyright 2026 Maolin Wang (modifications)
#
# This repository redistributes and modifies Apache-2.0 licensed components.
# Upstream attributions: see NOTICE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from wimsoslam.common import normalize_3d_coordinate


class DenseLayer(nn.Linear):
    """Single dense layer for MLP."""

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        """Resets dense layer parameters."""
        torch.nn.init.xavier_uniform_(self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class MLP_2D(nn.Module):
    """
    Simple MLP that takes in pixel information and outputs a pixel result.

    Args:
        name (str): name of this network.
        dim (int): input dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
    """

    def __init__(self, name="", dim=10, hidden_size=32, n_blocks=5, leaky=False):
        super().__init__()

        self.name = name
        self.no_grad_feature = False
        self.n_blocks = n_blocks

        self.linear = nn.ModuleList(
            [DenseLayer(dim, hidden_size, activation="relu")]
            + [DenseLayer(hidden_size, hidden_size, activation="relu") for i in range(n_blocks - 1)]
        )

        print(self.name)
        print(self.linear)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.output_linear = DenseLayer(hidden_size, 1, activation="linear")

        self.output_softplus = nn.Softplus()
        # self.output_sigmoid = nn.Sigmoid()

    def forward(self, features, **kwargs):
        for i, l in enumerate(self.linear):
            features = self.linear[i](features)
            features = F.relu(features)

        out = self.output_linear(features)
        out = self.output_softplus(out)
        # out = self.output_sigmoid(out)

        return out

class Decoders(nn.Module):
    """
    Decoders for SDF and RGB.
    Args:
        c_dim: feature dimensions
        hidden_size: hidden size of MLP
        truncation: truncation of SDF
        n_blocks: number of MLP blocks
        learnable_beta: whether to learn beta

    """
    def __init__(
            self,
            c_dim=32,
            num_sensors=1,
            conf_dim=50,
            col_conf_dim=75,
            hidden_size=16,
            truncation=0.08,
            n_blocks=2,
            learnable_beta=True
    ):
        super().__init__()

        self.num_sensors = num_sensors

        self.c_dim = c_dim
        self.truncation = truncation
        self.n_blocks = n_blocks

        ## layers for SDF decoder
        self.linears = nn.ModuleList(
            [nn.Linear(2 * c_dim, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])

        ## layers for RGB decoder
        self.c_linears = nn.ModuleList(
            [nn.Linear(2 * c_dim, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size)  for i in range(n_blocks - 1)])

        self.output_linear = nn.Linear(hidden_size, 1)
        self.c_output_linear = nn.Linear(hidden_size, 3)

        if learnable_beta:
            self.beta = nn.Parameter(10 * torch.ones(1))
        else:
            self.beta = 10

        self.color_conf_decoder = MLP_2D(name="c_var", dim=col_conf_dim, n_blocks=5, hidden_size=hidden_size)
        # self.color_conf_decoder = MLP_2D(name="c_var", dim=27, n_blocks=5, hidden_size=hidden_size)
        self.conf_decoders = []
        for i in range(num_sensors):
            self.conf_decoders.append(
                MLP_2D(
                    name=f"var{i}",
                    dim=conf_dim,
                    # dim=63,
                    n_blocks=5,
                    hidden_size=hidden_size,
                )
            )
        self.conf_decoders[-1] = self.conf_decoders[-1].to("cuda:0")


    def sample_plane_feature(self, p_nor, planes_xy, planes_xz, planes_yz):
        """
        Sample feature from planes
        Args:
            p_nor (tensor): normalized 3D coordinates
            planes_xy (list): xy planes
            planes_xz (list): xz planes
            planes_yz (list): yz planes
        Returns:
            feat (tensor): sampled features
        """
        vgrid = p_nor[None, :, None]

        feat = []
        for i in range(len(planes_xy)):
            xy = F.grid_sample(planes_xy[i], vgrid[..., [0, 1]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            xz = F.grid_sample(planes_xz[i], vgrid[..., [0, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            yz = F.grid_sample(planes_yz[i], vgrid[..., [1, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            feat.append(xy + xz + yz)
        feat = torch.cat(feat, dim=-1)

        return feat

    def get_raw_sdf(self, p_nor, all_planes):
        """
        Get raw SDF
        Args:
            p_nor (tensor): normalized 3D coordinates
            all_planes (Tuple): all feature planes
        Returns:
            sdf (tensor): raw SDF
        """
        planes_xy, planes_xz, planes_yz, c_planes_xy, c_planes_xz, c_planes_yz = all_planes
        feat = self.sample_plane_feature(p_nor, planes_xy, planes_xz, planes_yz)

        h = feat
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = F.relu(h, inplace=True)
        sdf = torch.tanh(self.output_linear(h)).squeeze()

        return sdf

    def get_raw_rgb(self, p_nor, all_planes):
        """
        Get raw RGB
        Args:
            p_nor (tensor): normalized 3D coordinates
            all_planes (Tuple): all feature planes
        Returns:
            rgb (tensor): raw RGB
        """
        planes_xy, planes_xz, planes_yz, c_planes_xy, c_planes_xz, c_planes_yz = all_planes
        c_feats = self.sample_plane_feature(p_nor, c_planes_xy, c_planes_xz, c_planes_yz)

        h = c_feats
        for i, l in enumerate(self.c_linears):
            h = self.c_linears[i](h)
            h = F.relu(h, inplace=True)
        rgb = torch.sigmoid(self.c_output_linear(h))

        return rgb

    def forward(self, p, all_planes, d_feats=None, c_feats=None, **kwargs):
        """
        Forward pass
        Args:
            p (tensor): 3D coordinates
            all_planes (Tuple): all feature planes
        Returns:
            raw (tensor): raw SDF and RGB
        """
        device = f"cuda:{p.get_device()}"
        p_shape = p.shape

        p_nor = normalize_3d_coordinate(p.clone(), self.bound)

        sdf = self.get_raw_sdf(p_nor, all_planes)
        rgb = self.get_raw_rgb(p_nor, all_planes)

        raw = torch.cat([rgb, sdf.unsqueeze(-1)], dim=-1)
        raw = raw.reshape(*p_shape[:-1], -1)

        if d_feats is not None:
            var = torch.zeros(d_feats.shape[0], self.num_sensors + 1).to(device).float()
            for i in range(self.num_sensors):
                var[..., i] = self.conf_decoders[i](d_feats[..., i]).squeeze()
            var[..., -1] = self.color_conf_decoder(c_feats).squeeze()
        else:
            var = torch.zeros(5000, self.num_sensors + 1).to(device).float()

        return (
            raw,
            var,
        )  # (N*samples,4) [color (3,), occupancy (1,)], (samples,2) [variance (1,)]
