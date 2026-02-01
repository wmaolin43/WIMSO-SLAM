# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 ams-OSRAM AG
# Copyright 2026 Maolin Wang (modifications)
#
# This repository redistributes and modifies Apache-2.0 licensed components.
# Upstream attributions: see NOTICE.

"""wimsoslam.system

EN:
    System orchestrator that allocates shared resources (planes, decoders, pose buffers)
    and launches tracking / mapping workers.

JP:
    共有リソース（plane / decoder / pose buffer）を用意し、
    Tracking と Mapping のワーカーを起動するオーケストレータ。

Notes / 注意:
    - This module focuses on clarity: configuration, shared memory, and process lifecycle.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.multiprocessing as mp

from wimsoslam import config as cfg_loader
from wimsoslam.Mapper import Mapper
from wimsoslam.Tracker import Tracker
from wimsoslam.utils.datasets import get_dataset
from wimsoslam.utils.Logger import Logger
from wimsoslam.utils.Mesher import Mesher
from wimsoslam.utils.Renderer import Renderer


torch.multiprocessing.set_sharing_strategy("file_system")


@dataclass
class OutputPaths:
    """Paths that are created under the output root.

    EN: Convenience container for output sub-directories.
    JP: 出力ディレクトリのサブパスをまとめるためのコンテナ。
    """

    root: str
    checkpoints: str
    meshes: str


class WIMSOSystem:
    """Main orchestration class.

    EN:
        Allocates shared tensors (feature planes, decoders, pose buffers) and
        dispatches tracking and mapping worker processes.

    JP:
        共有テンソル（feature plane、decoder、姿勢バッファ）を確保し、
        Tracking と Mapping のプロセスを起動します。

    Parameters / 引数:
        cfg_dict: Parsed YAML configuration.
        cli_args: Arguments from CLI (may override paths).
    """

    def __init__(self, cfg_dict: dict, cli_args):
        self.cfg = cfg_dict
        self.args = cli_args

        self.verbose: bool = bool(cfg_dict.get("verbose", False))
        self.device: str = str(cfg_dict.get("device", "cuda:0"))
        self.dataset_name: str = str(cfg_dict.get("dataset"))
        self.truncation: float = float(cfg_dict["model"]["truncation"])
        self.scale: float = float(cfg_dict.get("scale", 1.0))

        self.paths = self._prepare_output_dirs(cfg_dict, cli_args)

        # Camera intrinsics (may be changed by preprocessing settings)
        cam = cfg_dict["cam"]
        self.H = int(cam["H"])
        self.W = int(cam["W"])
        self.fx = float(cam["fx"])
        self.fy = float(cam["fy"])
        self.cx = float(cam["cx"])
        self.cy = float(cam["cy"])
        self._apply_cam_preprocessing()

        # Model, bounds, planes (shared across processes)
        self.shared_decoders = cfg_loader.get_model(cfg_dict)
        self._load_and_set_bounds(cfg_dict)
        self._init_feature_planes(cfg_dict)

        # Multiprocessing: spawn is safer with CUDA.
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        # Dataset / shared pose buffers
        self.frame_reader = get_dataset(cfg_dict, cli_args, self.scale)
        self.num_frames = len(self.frame_reader)

        self.est_c2w = torch.zeros((self.num_frames, 4, 4), device=self.device)
        self.est_c2w.share_memory_()

        self.gt_c2w = torch.zeros((self.num_frames, 4, 4))
        self.gt_c2w.share_memory_()

        # Shared counters / flags
        self.shared_idx = torch.zeros((1,), dtype=torch.int32)
        self.shared_idx.share_memory_()

        self.mapping_first_frame = torch.zeros((1,), dtype=torch.int32)
        self.mapping_first_frame.share_memory_()

        self.mapping_frame_idx = torch.zeros((1,), dtype=torch.int32)
        self.mapping_frame_idx.share_memory_()

        self.mapping_step_counter = torch.zeros((1,), dtype=torch.int32)
        self.mapping_step_counter.share_memory_()

        # -----------------------------------------------------------------
        # Backward-compatible attribute names (for upstream worker modules)
        # EN: Keep older names so Mapper/Tracker remain functional.
        # JP: 既存の Mapper/Tracker を動かすため、従来名も用意します。
        # -----------------------------------------------------------------
        self.output = self.paths.root
        self.idx = self.shared_idx
        self.mapping_idx = self.mapping_frame_idx
        self.mapping_cnt = self.mapping_step_counter
        self.gt_c2w_list = self.gt_c2w
        self.estimate_c2w_list = self.est_c2w

        # -----------------------------------------------------------------


        # Move planes / decoders to device & share
        self._share_planes_and_decoders()

        # Utilities & workers
        self.renderer = Renderer(cfg_dict, self)
        self.mesher = Mesher(cfg_dict, cli_args, self)
        self.logger = Logger(self)
        self.mapper = Mapper(cfg_dict, cli_args, self)
        self.tracker = Tracker(cfg_dict, cli_args, self)

        self._print_paths_summary()

    # ---------------------------------------------------------------------
    # Setup helpers
    # ---------------------------------------------------------------------

    def _prepare_output_dirs(self, cfg_dict: dict, cli_args) -> OutputPaths:
        """Create output directories.

        EN: Uses CLI overrides when provided.
        JP: CLI の指定があればそれを優先して出力ディレクトリを作成します。
        """

        out_root = cli_args.output if getattr(cli_args, "output", None) else cfg_dict["data"]["output"]
        ckpt_dir = os.path.join(out_root, "ckpts")
        mesh_dir = os.path.join(out_root, "mesh")

        os.makedirs(out_root, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(mesh_dir, exist_ok=True)

        # keep compatibility with existing scripts
        os.makedirs(os.path.join(out_root, "tracking_vis"), exist_ok=True)
        os.makedirs(os.path.join(out_root, "mapping_vis"), exist_ok=True)

        return OutputPaths(root=out_root, checkpoints=ckpt_dir, meshes=mesh_dir)

    def _apply_cam_preprocessing(self) -> None:
        """Update intrinsics after preprocessing.

        EN:
            Some datasets are resized/cropped before use. Those ops effectively
            change intrinsics (fx, fy, cx, cy) and image size (H, W).

        JP:
            入力画像を resize / crop する場合、内部パラメータ（fx, fy, cx, cy）と
            画像サイズ（H, W）を更新する必要があります。
        """

        cam_cfg = self.cfg["cam"]

        # Resize (lietorch naming: crop_size)
        if "crop_size" in cam_cfg:
            crop_h, crop_w = cam_cfg["crop_size"]
            sx = crop_w / self.W
            sy = crop_h / self.H
            self.fx *= sx
            self.fy *= sy
            self.cx *= sx
            self.cy *= sy
            self.W = int(crop_w)
            self.H = int(crop_h)

        # Crop edge
        edge = int(cam_cfg.get("crop_edge", 0))
        if edge > 0:
            self.H -= edge * 2
            self.W -= edge * 2
            self.cx -= edge
            self.cy -= edge

    def _load_and_set_bounds(self, cfg_dict: dict) -> None:
        """Load scene bounds and propagate to decoders.

        EN: Bounds are scaled by `scale` and expanded to be divisible.
        JP: Bound は `scale` に合わせてスケールし、分割可能になるよう拡張します。
        """

        raw_bound = np.array(cfg_dict["mapping"]["bound"], dtype=np.float32)
        bound = torch.from_numpy(raw_bound * self.scale).float()

        divisible = float(cfg_dict["planes_res"]["bound_dividable"])
        # enlarge upper bound to be divisible
        bound[:, 1] = (((bound[:, 1] - bound[:, 0]) / divisible).int() + 1) * divisible + bound[:, 0]

        self.bound = bound
        self.shared_decoders.bound = self.bound

    def _init_feature_planes(self, cfg_dict: dict) -> None:
        """Initialize multi-resolution feature planes.

        EN:
            Planes are initialized with small Gaussian noise.
            Two resolutions are used for geometry (coarse/fine) and color.

        JP:
            Feature plane は小さいガウスノイズで初期化します。
            Geometry と Color で coarse/fine の2解像度を使用します。
        """

        planes_res = [cfg_dict["planes_res"]["coarse"], cfg_dict["planes_res"]["fine"]]
        color_res = [cfg_dict["c_planes_res"]["coarse"], cfg_dict["c_planes_res"]["fine"]]

        feat_dim = int(cfg_dict["model"]["c_dim"])
        extent = (self.bound[:, 1] - self.bound[:, 0]).tolist()

        def _make_planes(res_list):
            xy, xz, yz = [], [], []
            for res in res_list:
                grid = list(map(int, (np.array(extent) / float(res)).tolist()))
                # swap x/z for memory layout consistency
                grid[0], grid[2] = grid[2], grid[0]
                xy.append(torch.empty((1, feat_dim, grid[1], grid[2])).normal_(mean=0.0, std=0.01))
                xz.append(torch.empty((1, feat_dim, grid[0], grid[2])).normal_(mean=0.0, std=0.01))
                yz.append(torch.empty((1, feat_dim, grid[0], grid[1])).normal_(mean=0.0, std=0.01))
            return xy, xz, yz

        self.shared_planes_xy, self.shared_planes_xz, self.shared_planes_yz = _make_planes(planes_res)
        self.shared_c_planes_xy, self.shared_c_planes_xz, self.shared_c_planes_yz = _make_planes(color_res)

    def _share_planes_and_decoders(self) -> None:
        """Move shared tensors to device and enable shared memory.

        EN: This is required so tracking & mapping processes see consistent parameters.
        JP: Tracking/Mapping で同じパラメータを参照するために shared memory を使います。
        """

        for plane_bank in (self.shared_planes_xy, self.shared_planes_xz, self.shared_planes_yz):
            for i, p in enumerate(plane_bank):
                p = p.to(self.device)
                p.share_memory_()
                plane_bank[i] = p

        for plane_bank in (self.shared_c_planes_xy, self.shared_c_planes_xz, self.shared_c_planes_yz):
            for i, p in enumerate(plane_bank):
                p = p.to(self.device)
                p.share_memory_()
                plane_bank[i] = p

        self.shared_decoders = self.shared_decoders.to(self.device)
        self.shared_decoders.share_memory()

    def _print_paths_summary(self) -> None:
        print(f"INFO: Output root: {self.paths.root}")
        print(f"INFO: Mesh outputs: {self.paths.meshes}")
        print(f"INFO: Checkpoints: {self.paths.checkpoints}")

    # ---------------------------------------------------------------------
    # Process entrypoints
    # ---------------------------------------------------------------------

    def _tracking_process(self, rank: int) -> None:
        """Tracking worker.

        EN: Waits until the first mapping step is done, then starts tracking.
        JP: 最初の Mapping が終わるまで待ってから Tracking を開始します。
        """

        while True:
            if int(self.mapping_first_frame[0].item()) == 1:
                break
            time.sleep(1.0)

        self.tracker.run()

    def _mapping_process(self, rank: int) -> None:
        """Mapping worker.

        EN: Immediately starts mapping.
        JP: Mapping は直ちに開始します。
        """

        self.mapper.run()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def run(self) -> None:
        """Launch tracking & mapping.

        EN: Spawns two child processes.
        JP: 2つの子プロセスを起動します。
        """

        workers = []
        for rank, fn in enumerate((self._tracking_process, self._mapping_process)):
            proc = mp.Process(target=fn, args=(rank,))
            proc.start()
            workers.append(proc)

        for proc in workers:
            proc.join()


# Multiprocessing safe-guard
if __name__ == "__main__":
    pass
