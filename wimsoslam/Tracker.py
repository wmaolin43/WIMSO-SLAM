# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 ams-OSRAM AG
# Copyright 2026 Maolin Wang (modifications)
#
# This repository redistributes and modifies Apache-2.0 licensed components.
# Upstream attributions: see NOTICE.

"""wimsoslam.Tracker

EN:
    Camera tracking worker.
    It optimizes per-frame camera pose by minimizing a weighted combination of:
      - SDF supervision around the observed surface
      - Depth reprojection error (uncertainty-aware)
      - Color reconstruction error

JP:
    カメラ追跡ワーカー。
    各フレームのカメラ姿勢を以下の損失の重み付き和で最適化します:
      - 観測表面周辺の SDF 监督
      - 深度誤差（uncertainty を用いた重み付け）
      - 色の再構成誤差

Notes / 注意:
    - The mapping thread updates shared planes/decoders. The tracker periodically
      pulls them into local copies.
    - Multi-depth-sensor inputs are supported via prefixed keys ("2_depth", "3_depth", ...).
"""

from __future__ import annotations

import copy
import os
import time
from typing import Dict, List, Tuple

import torch
from colorama import Fore, Style
from torch.utils.data import DataLoader
from tqdm import tqdm

from wimsoslam.common import matrix_to_cam_pose, cam_pose_to_matrix, get_samples
from wimsoslam.utils.datasets import get_dataset
from wimsoslam.utils.Frame_Visualizer import Frame_Visualizer


def _sensor_prefix(sensor_index: int) -> str:
    """Return the feature-key prefix for a sensor.

    EN: Primary sensor uses empty prefix, secondary sensors use '2_', '3_', ...
    JP: 主センサーは空文字、追加センサーは '2_', '3_', ... を使用します。
    """

    return "" if sensor_index == 0 else f"{sensor_index + 1}_"


class PoseTracker:
    """Tracking worker.

    EN: Optimizes camera pose for each incoming frame.
    JP: 各フレームに対してカメラ姿勢を最適化します。
    """

    def __init__(self, cfg: dict, cli_args, slam_system):
        self.cfg = cfg
        self.args = cli_args
        self.slam = slam_system

        # Sensor/feature configuration
        self.num_sensors = int(cfg.get("num_sensors", 1))
        self.patch_size = int(cfg.get("patch_size", 1))
        self.sensor_prefixes = [_sensor_prefix(i) for i in range(self.num_sensors)]

        # Features used in tracking (must exist in dataset output)
        base_features = ["color", "local"]
        depth_features = ["depth", "normal", "error", "angle", "dx", "dy"]
        self.feature_keys: List[str] = base_features + [f"{p}{k}" for p in self.sensor_prefixes for k in depth_features]

        # Short aliases to shared state
        self.idx = slam_system.idx
        self.bound = slam_system.bound
        self.output_dir = slam_system.output
        self.verbose = slam_system.verbose
        self.renderer = slam_system.renderer
        self.gt_c2w_list = slam_system.gt_c2w_list
        self.mapping_idx = slam_system.mapping_idx
        self.mapping_cnt = slam_system.mapping_cnt
        self.shared_decoders = slam_system.shared_decoders
        self.estimate_c2w_list = slam_system.estimate_c2w_list
        self.truncation = slam_system.truncation

        # Shared planes (geometry + color)
        self.shared_planes_xy = slam_system.shared_planes_xy
        self.shared_planes_xz = slam_system.shared_planes_xz
        self.shared_planes_yz = slam_system.shared_planes_yz
        self.shared_c_planes_xy = slam_system.shared_c_planes_xy
        self.shared_c_planes_xz = slam_system.shared_c_planes_xz
        self.shared_c_planes_yz = slam_system.shared_c_planes_yz

        # Tracking hyperparameters
        tr_cfg = cfg["tracking"]
        self.lr_trans = float(tr_cfg["lr_T"])
        self.lr_rot = float(tr_cfg["lr_R"])
        self.device = cfg["device"]
        self.num_iters = int(tr_cfg["iters"])
        self.use_gt_pose = bool(tr_cfg["gt_camera"])
        self.sample_rays = int(tr_cfg["pixels"])

        self.w_sdf_free = float(tr_cfg["w_sdf_fs"])
        self.w_sdf_center = float(tr_cfg["w_sdf_center"])
        self.w_sdf_tail = float(tr_cfg["w_sdf_tail"])
        self.w_depth = float(tr_cfg["w_depth"])
        self.w_color = float(tr_cfg["w_color"])

        self.ignore_edge_W = int(tr_cfg["ignore_edge_W"])
        self.ignore_edge_H = int(tr_cfg["ignore_edge_H"])
        self.const_speed = bool(tr_cfg["const_speed_assumption"])
        self.no_vis_on_first = bool(tr_cfg["no_vis_on_first_frame"])

        self.every_frame = int(cfg["mapping"]["every_frame"])

        # Dataset
        self.scale = float(cfg["scale"])
        self.frame_reader = get_dataset(cfg, cli_args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(
            self.frame_reader,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
            prefetch_factor=2,
        )

        # Visualization
        vis_dir = os.path.join(self.output_dir, "tracking_vis")
        self.visualizer = Frame_Visualizer(
            freq=int(tr_cfg["vis_freq"]),
            inside_freq=int(tr_cfg["vis_inside_freq"]),
            vis_dir=vis_dir,
            renderer=self.renderer,
            truncation=self.truncation,
            verbose=self.verbose,
            device=self.device,
        )

        # Camera intrinsics
        self.H, self.W = slam_system.H, slam_system.W
        self.fx, self.fy = slam_system.fx, slam_system.fy
        self.cx, self.cy = slam_system.cx, slam_system.cy

        # Local copies of shared parameters (not trainable in tracking)
        self.decoders = copy.deepcopy(self.shared_decoders)
        self.planes_xy = copy.deepcopy(self.shared_planes_xy)
        self.planes_xz = copy.deepcopy(self.shared_planes_xz)
        self.planes_yz = copy.deepcopy(self.shared_planes_yz)
        self.c_planes_xy = copy.deepcopy(self.shared_c_planes_xy)
        self.c_planes_xz = copy.deepcopy(self.shared_c_planes_xz)
        self.c_planes_yz = copy.deepcopy(self.shared_c_planes_yz)

        for p in self.decoders.parameters():
            p.requires_grad_(False)

        self._last_synced_mapping_idx = -1

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def _sdf_supervision(self, sdf: torch.Tensor, z_vals: torch.Tensor, gt_depth: torch.Tensor) -> torch.Tensor:
        """Compute WIMSO-SLAM-style SDF loss along rays.

        EN:
            Penalizes SDF in free space and enforces consistency near the observed surface.
        JP:
            Free-space での SDF と、表面近傍での整合性を強制する損失です。
        """

        trunc = float(self.truncation)

        front = (z_vals < (gt_depth[:, None] - trunc))
        back = (z_vals > (gt_depth[:, None] + trunc))
        center = (z_vals > (gt_depth[:, None] - 0.4 * trunc)) & (z_vals < (gt_depth[:, None] + 0.4 * trunc))
        tail = (~front) & (~back) & (~center)

        fs_loss = torch.mean((sdf[front] - 1.0) ** 2)
        center_loss = torch.mean(((z_vals + sdf * trunc)[center] - gt_depth[:, None].expand_as(z_vals)[center]) ** 2)
        tail_loss = torch.mean(((z_vals + sdf * trunc)[tail] - gt_depth[:, None].expand_as(z_vals)[tail]) ** 2)

        return self.w_sdf_free * fs_loss + self.w_sdf_center * center_loss + self.w_sdf_tail * tail_loss

    # ------------------------------------------------------------------
    # Sync from mapping
    # ------------------------------------------------------------------

    def _sync_from_mapper(self) -> None:
        """Pull updated planes/decoders from the mapping thread.

        EN: This keeps tracking consistent with the current map.
        JP: Mapping 側の最新パラメータを tracking に反映します。
        """

        if int(self.mapping_idx[0].item()) == int(self._last_synced_mapping_idx):
            return

        if self.verbose:
            print("Tracking: syncing parameters from mapping")

        self.decoders.load_state_dict(self.shared_decoders.state_dict())

        for shared_bank, local_bank in zip(
            [self.shared_planes_xy, self.shared_planes_xz, self.shared_planes_yz],
            [self.planes_xy, self.planes_xz, self.planes_yz],
        ):
            for i, plane in enumerate(shared_bank):
                local_bank[i] = plane.detach()

        for shared_bank, local_bank in zip(
            [self.shared_c_planes_xy, self.shared_c_planes_xz, self.shared_c_planes_yz],
            [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz],
        ):
            for i, plane in enumerate(shared_bank):
                local_bank[i] = plane.detach()

        self._last_synced_mapping_idx = self.mapping_idx[0].clone()

    # ------------------------------------------------------------------
    # Optimization step
    # ------------------------------------------------------------------

    def _tracking_step(self, cam_pose: torch.Tensor, frame_pack: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
        """One gradient step for pose parameters.

        EN: Samples rays, renders predictions, computes loss, backprops, and updates pose.
        JP: レイをサンプリング→レンダリング→損失計算→逆伝播→姿勢更新、を 1 ステップ行います。
        """

        device = self.device

        # Build c2w from camera parameters
        c2w = cam_pose_to_matrix(cam_pose)
        frame_pack["est_c2w"] = c2w

        # Prepare dict on device
        optim_pack = {k: frame_pack[k].to(device) for k in (self.feature_keys + ["est_c2w"]) if k in frame_pack}

        # Sample rays + features
        rays_o, rays_d, sample_dict = get_samples(
            self.ignore_edge_H,
            self.H - self.ignore_edge_H,
            self.ignore_edge_W,
            self.W - self.ignore_edge_W,
            self.sample_rays,
            self.H,
            self.W,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            optim_pack,
            self.feature_keys,
            device,
            kernel=self.patch_size,
        )

        sample_dict["rays_o"] = rays_o.float()
        sample_dict["rays_d"] = rays_d.float()

        # Filter rays that are outside the map volume
        with torch.no_grad():
            ro = sample_dict["rays_o"].detach().unsqueeze(-1)  # (N, 3, 1)
            rd = sample_dict["rays_d"].detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.bound.unsqueeze(0).to(device) - ro) / rd
            t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            inside_mask = t >= sample_dict.get("depth", torch.zeros_like(t))

        # If no rays survive, return a large loss to keep optimizer stable
        if int(inside_mask.sum().item()) < 8:
            return float("inf")

        for k, v in list(sample_dict.items()):
            if torch.is_tensor(v) and v.shape[0] == inside_mask.shape[0]:
                sample_dict[k] = v[inside_mask]

        # Render
        planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)
        pred_depth, pred_color, sdf, z_vals, var, uncert = self.renderer.render_batch_ray(
            planes,
            self.decoders,
            sample_dict["rays_d"],
            sample_dict["rays_o"],
            sample_dict,
            self.device,
            self.truncation,
            gt_depth=sample_dict.get("depth", None),
        )

        # Detach uncertainty heads in tracking (pose-only optimization)
        uncert = uncert.detach()
        var = var.detach()

        # Build loss
        total_sdf = 0.0
        total_depth = 0.0
        used = 0

        for i, prefix in enumerate(self.sensor_prefixes):
            depth_key = f"{prefix}depth"
            if depth_key not in sample_dict:
                continue
            gt_d = sample_dict[depth_key]

            denom = torch.sqrt(uncert + 1e-10) + var[:, i]
            depth_resid = torch.abs(gt_d - pred_depth) / denom
            valid = (gt_d > 0) & (depth_resid < 10.0 * depth_resid.median())

            if int(valid.sum().item()) < 8:
                continue

            total_sdf = total_sdf + self._sdf_supervision(sdf[valid], z_vals[valid], gt_d[valid])
            total_depth = total_depth + (torch.square(gt_d[valid] - pred_depth[valid]) / denom[valid]).mean()
            used += 1

        if used > 0:
            total_sdf = total_sdf / float(used)
            total_depth = total_depth / float(used)

        # Color loss (uses rgb variance head in var[:, -1])
        rgb_var = var[:, -1].unsqueeze(-1).repeat(1, 3)
        rgb_valid = sample_dict.get("color", None)
        if rgb_valid is not None:
            color_err = torch.square(sample_dict["color"] - pred_color)
            color_loss = (color_err / (1e-3 + rgb_var)).mean()
        else:
            color_loss = torch.tensor(0.0, device=device)

        loss = total_sdf + self.w_depth * total_depth + self.w_color * color_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        return float(loss.item())

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main tracking loop / トラッキング本体."""

        device = self.device

        iterator = self.frame_loader if self.verbose else tqdm(self.frame_loader, smoothing=0.05)

        # Local caches
        prev_c2w = None

        for frame_bundle, idx_tensor, gt_color, gt_depth, gt_c2w in iterator:
            idx = int(idx_tensor[0].item())

            # Unpack frame dict
            frame_pack = {k: frame_bundle[k][0] for k in frame_bundle}

            gt_color = gt_color.to(device, non_blocking=True)
            gt_depth = gt_depth.to(device, non_blocking=True)
            gt_c2w = gt_c2w.to(device, non_blocking=True)

            if not self.verbose:
                iterator.set_description(f"Tracking Frame {idx}")

            # Wait until mapping is ready for the previous frame (when mapping runs every N frames)
            if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                while int(self.mapping_idx[0].item()) != idx - 1:
                    time.sleep(0.001)
                prev_c2w = self.estimate_c2w_list[idx - 1].unsqueeze(0).to(device)

            self._sync_from_mapper()

            if self.verbose:
                print(Fore.MAGENTA + f"Tracking Frame {idx}" + Style.RESET_ALL)

            if idx == 0 or self.use_gt_pose:
                c2w = gt_c2w
                if not self.no_vis_on_first:
                    planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)
                    self.visualizer.save_imgs(idx, 0, frame_pack, self.feature_keys, gt_depth, gt_color, c2w.squeeze(), planes, self.decoders)

            else:
                # Initialize pose
                if self.const_speed and idx - 2 >= 0 and prev_c2w is not None:
                    pre = torch.stack([self.estimate_c2w_list[idx - 2], prev_c2w.squeeze(0)], dim=0)
                    pre_pose = matrix_to_cam_pose(pre)
                    init_pose = 2 * pre_pose[1:] - pre_pose[0:1]
                else:
                    init_pose = matrix_to_cam_pose(prev_c2w)

                # Optimize pose parameters (R quaternion + t)
                t_param = torch.nn.Parameter(init_pose[:, -3:].clone())
                r_param = torch.nn.Parameter(init_pose[:, :4].clone())

                optimizer = torch.optim.Adam(
                    [
                        {"params": [t_param], "lr": self.lr_trans, "betas": (0.5, 0.999)},
                        {"params": [r_param], "lr": self.lr_rot, "betas": (0.5, 0.999)},
                    ]
                )

                best_loss = float("inf")
                best_pose = None

                for it in range(self.num_iters):
                    cam_pose = torch.cat([r_param, t_param], dim=-1)

                    planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)
                    self.visualizer.save_imgs(idx, it, frame_pack, self.feature_keys, gt_depth, gt_color, cam_pose, planes, self.decoders)

                    loss_val = self._tracking_step(cam_pose, frame_pack, optimizer)
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_pose = cam_pose.detach().clone()

                c2w = cam_pose_to_matrix(best_pose)
                frame_pack["est_c2w"] = c2w

            # Commit results to shared buffers
            frame_index = int(frame_pack.get("idx", idx))
            self.estimate_c2w_list[frame_index] = c2w.squeeze(0).clone()
            self.gt_c2w_list[frame_index] = gt_c2w.squeeze(0).clone()

            prev_c2w = c2w.clone()
            self.idx[0] = frame_index


# Backward-compatible name
Tracker = PoseTracker
