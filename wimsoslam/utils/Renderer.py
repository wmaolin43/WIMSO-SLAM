# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 ams-OSRAM AG
# Copyright 2026 Maolin Wang (modifications)
#
# This repository redistributes and modifies Apache-2.0 licensed components.
# Upstream attributions: see NOTICE.

"""wimsoslam.utils.Renderer

EN:
    Ray renderer used by tracking and mapping.
    It performs stratified sampling + importance resampling, evaluates the SDF/RGB
    decoders, and integrates along rays to produce depth, color, and uncertainty.

JP:
    Tracking / Mapping で使用するレイレンダラ。
    Stratified sampling と importance resampling を行い、SDF/RGB decoder を評価して
    レイ積分により depth / color / uncertainty を算出します。

Design notes / 設計メモ:
    - This renderer also forwards per-ray *auxiliary features* (e.g., depth error,
      normals, gradients, angle) to the confidence decoders.
    - Batch dictionaries can optionally contain multiple depth sensors. Keys are
      encoded as prefixes (e.g., "2_depth", "2_error", ...).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch

from wimsoslam.common import get_rays, sample_pdf, normalize_3d_coordinate


class Renderer:
    """Renderer for WIMSO-SLAM.

    EN:
        Provides three primary utilities:
        - `render_batch_ray`: render a batch of rays (used in both tracking & mapping)
        - `render_img`: render a full image by chunking
        - `render`: convenience wrapper to build rays from a pose

    JP:
        主な機能:
        - `render_batch_ray`: レイバッチのレンダリング（Tracking/Mapping共通）
        - `render_img`: 画像全体をチャンク分割してレンダリング
        - `render`: 姿勢からレイを生成してレンダリング
    """

    def __init__(self, cfg: dict, slam_system):
        self.cfg = cfg
        self.slam = slam_system

        # Sampling config
        self.n_stratified = int(cfg['rendering']['n_stratified'])
        self.n_importance = int(cfg['rendering']['n_importance'])
        self.n_rays = int(cfg['rendering']['n_rays'])

        # Coordinate normalization bound
        self.bound = slam_system.bound.to(slam_system.device)

        # Uncertainty configuration
        self.num_sensors = int(cfg.get('num_sensors', 1))
        self.num_features = int(cfg.get('num_features', 2))
        self.patch_size = int(cfg.get('patch_size', 1))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sorted_feature_keys(batch_dict: Dict[str, torch.Tensor]) -> List[str]:
        """Collect feature keys in a deterministic order.

        EN:
            We exclude meta keys (rays, coords, images) and keep per-ray feature keys.
        JP:
            rays や座標などのメタ情報を除外し、特徴量キーのみを安定順序で取得します。
        """

        ignore = {'rays_o', 'rays_d', 'target_s', 'target_d', 'target_dsurf', 'target_depth',
                  'gt_depth', 'gt_color', 'first', 'frame_idx', 'stage', 'inds'}
        keys = [k for k in batch_dict.keys() if k not in ignore and not k.endswith('_coords')]
        keys.sort()
        return keys

    def _split_feature_keys(self, keys: List[str]) -> Tuple[List[str], List[str]]:
        """Split keys into depth-related and color-related feature keys.

        EN:
            Current implementation uses the substring 'color' to identify the color cue.
        JP:
            現実装では 'color' を含むかどうかで色関連特徴を判定します。
        """

        depth_keys, color_keys = [], []
        for k in keys:
            if 'color' in k:
                color_keys.append(k)
            else:
                depth_keys.append(k)
        return depth_keys, color_keys

    # ------------------------------------------------------------------
    # Core rendering
    # ------------------------------------------------------------------

    def render_batch_ray(
        self,
        all_planes,
        decoders,
        rays_d: torch.Tensor,
        rays_o: torch.Tensor,
        batch_dict: Dict[str, torch.Tensor],
        device: str,
        truncation: float,
        gt_depth: Optional[torch.Tensor] = None,
    ):
        """Render a batch of rays.

        EN:
            Performs:
            1) stratified sampling in depth
            2) evaluates SDF/RGB (+ variance heads when enabled)
            3) computes weights, integrates depth/color
            4) optional importance sampling refinement

        JP:
            以下を実行します:
            1) 深度方向の stratified sampling
            2) SDF/RGB の評価（有効なら分散ヘッドも）
            3) 重み計算と積分（depth/color）
            4) importance sampling による再サンプル（任意）

        Returns / 戻り値:
            depth, color, sdf, z_vals, var, model_uncert
        """

        # --------------------------------------------------------------
        # 1) Sampling
        # --------------------------------------------------------------
        n_rays = rays_o.shape[0]

        if gt_depth is None:
            # Scene-centric sampling range
            t_near = 0.0
            t_far = float(self.cfg['rendering']['far'])
        else:
            # Around GT depth
            t_near = gt_depth * 0.8
            t_far = gt_depth * 1.2

        z_vals = torch.linspace(0.0, 1.0, steps=self.n_stratified, device=device)
        if gt_depth is None:
            z_vals = t_near * (1.0 - z_vals) + t_far * z_vals
            z_vals = z_vals.expand(n_rays, self.n_stratified)
        else:
            z_vals = t_near * (1.0 - z_vals) + t_far * z_vals

        # Add stratified noise
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        # --------------------------------------------------------------
        # 2) Decoder evaluation
        # --------------------------------------------------------------
        # Depth-related features may come from multiple sensors.
        feature_keys = self._sorted_feature_keys(batch_dict)
        depth_keys, color_keys = self._split_feature_keys(feature_keys)

        # Collect per-ray auxiliary features in a (N_rays, F, num_sensors) tensor.
        # We infer the sensor index from key prefixes like "2_depth".
        d_feat_buckets: List[List[torch.Tensor]] = [[] for _ in range(self.num_sensors)]
        c_feat_list: List[torch.Tensor] = []

        for k in depth_keys:
            # Sensor prefix convention: "{idx}_xxx". If no prefix, assume sensor 0.
            if '_' in k and k.split('_', 1)[0].isdigit():
                sensor_id = int(k.split('_', 1)[0])
                base_key = k.split('_', 1)[1]
            else:
                sensor_id = 0
                base_key = k

            if base_key == 'depth':
                # Skip depth itself as confidence feature unless desired.
                pass

            if sensor_id < self.num_sensors:
                d_feat_buckets[sensor_id].append(batch_dict[k])

        # Color cue features (used for color confidence head)
        for k in color_keys:
            c_feat_list.append(batch_dict[k])

        # If nothing was provided, keep empty tensors to satisfy decoder API.
        if any(len(b) > 0 for b in d_feat_buckets):
            d_feats_list = [torch.cat(b, dim=-1) if len(b) > 0 else torch.zeros((n_rays, 0), device=device)
                            for b in d_feat_buckets]
            d_feats = torch.stack(d_feats_list, dim=-1)  # (N_rays, F, num_sensors)
        else:
            d_feats = None

        c_feats = torch.cat(c_feat_list, dim=-1) if len(c_feat_list) > 0 else None

        raw, var = decoders(pts, all_planes, d_feats, c_feats)

        rgb = raw[..., :3]
        sdf = raw[..., 3]

        # --------------------------------------------------------------
        # 3) Integrate
        # --------------------------------------------------------------
        # Convert SDF to alpha weights (truncated SDF)
        if gt_depth is None:
            # occupancy-like field
            sigma = torch.sigmoid(-sdf / truncation) * 10.0
        else:
            # around GT depth we can be sharper
            sigma = torch.sigmoid(-sdf / truncation) * 10.0

        # Distance between consecutive samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

        alpha = 1.0 - torch.exp(-sigma * dists)
        trans = torch.cumprod(torch.cat([torch.ones((n_rays, 1), device=device), 1.0 - alpha + 1e-10], -1), -1)[:, :-1]
        weights = alpha * trans

        depth = torch.sum(weights * z_vals, dim=-1)
        color = torch.sum(weights[..., None] * rgb, dim=-2)

        # model uncertainty: entropy-like term derived from weights distribution
        w_sum = weights.sum(dim=-1) + 1e-8
        probs = weights / w_sum.unsqueeze(-1)
        model_uncert = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

        # --------------------------------------------------------------
        # 4) Importance resampling (optional)
        # --------------------------------------------------------------
        if self.n_importance > 0:
            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_import = sample_pdf(z_vals_mid, weights[..., 1:-1], self.n_importance, det=False)
            z_import = z_import.detach()
            z_all, _ = torch.sort(torch.cat([z_vals, z_import], dim=-1), dim=-1)

            pts_fine = rays_o[..., None, :] + rays_d[..., None, :] * z_all[..., :, None]
            raw_fine, var = decoders(pts_fine, all_planes, d_feats, c_feats)

            rgb_fine = raw_fine[..., :3]
            sdf_fine = raw_fine[..., 3]

            sigma_fine = torch.sigmoid(-sdf_fine / truncation) * 10.0
            dists_f = z_all[..., 1:] - z_all[..., :-1]
            dists_f = torch.cat([dists_f, 1e10 * torch.ones_like(dists_f[..., :1])], dim=-1)

            alpha_f = 1.0 - torch.exp(-sigma_fine * dists_f)
            trans_f = torch.cumprod(torch.cat([torch.ones((n_rays, 1), device=device), 1.0 - alpha_f + 1e-10], -1), -1)[:, :-1]
            weights_f = alpha_f * trans_f

            depth = torch.sum(weights_f * z_all, dim=-1)
            color = torch.sum(weights_f[..., None] * rgb_fine, dim=-2)

            w_sum = weights_f.sum(dim=-1) + 1e-8
            probs = weights_f / w_sum.unsqueeze(-1)
            model_uncert = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

            sdf = sdf_fine
            z_vals = z_all

        return depth, color, sdf, z_vals, var, model_uncert

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def render_img(self, all_planes, decoders, c2w, device, truncation, gt_depth=None):
        """Render a full image.

        EN: Rays are chunked to avoid OOM.
        JP: OOM 回避のため、レイを分割して処理します。
        """

        H, W = self.slam.H, self.slam.W
        rays_o, rays_d = get_rays(H, W, self.slam.fx, self.slam.fy, self.slam.cx, self.slam.cy, c2w, device)

        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        if gt_depth is not None:
            gt_depth = gt_depth.reshape(-1)

        depths, colors = [], []
        sdf_all, z_all, var_all, u_all = [], [], [], []

        chunk = int(self.cfg['rendering'].get('chunk', 4096))
        for st in range(0, rays_o.shape[0], chunk):
            ed = min(st + chunk, rays_o.shape[0])
            batch_o = rays_o[st:ed]
            batch_d = rays_d[st:ed]
            batch_depth = gt_depth[st:ed] if gt_depth is not None else None

            # minimal batch_dict: no extra features for pure rendering
            tmp_dict = {'rays_o': batch_o, 'rays_d': batch_d}
            d, c, sdf, z, v, u = self.render_batch_ray(all_planes, decoders, batch_d, batch_o, tmp_dict, device, truncation, batch_depth)

            depths.append(d)
            colors.append(c)
            sdf_all.append(sdf)
            z_all.append(z)
            var_all.append(v)
            u_all.append(u)

        depth_img = torch.cat(depths, dim=0).reshape(H, W)
        color_img = torch.cat(colors, dim=0).reshape(H, W, 3)

        return depth_img, color_img

    def render(self, all_planes, decoders, c2w, device, truncation, gt_depth=None):
        """Alias for render_img for backward compatibility.

        EN: Older code called this method for full-image rendering.
        JP: 互換のためのエイリアスです。
        """

        return self.render_img(all_planes, decoders, c2w, device, truncation, gt_depth)
