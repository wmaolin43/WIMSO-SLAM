# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 ams-OSRAM AG
# Copyright 2026 Maolin Wang (modifications)
#
# This repository redistributes and modifies Apache-2.0 licensed components.
# Upstream attributions: see NOTICE.

"""wimsoslam.utils.datasets

EN:
    Dataset loaders and pre-processing utilities.

    This module provides PyTorch-style datasets for common RGB-D SLAM benchmarks
    (e.g., Replica, TUM RGB-D, ScanNet, Co-Fusion), and returns each frame as a
    dictionary of tensors (color, depth, and optional auxiliary channels).

JP:
    データセット読み込み・前処理ユーティリティ。

    Replica / TUM RGB-D / ScanNet / Co-Fusion などのRGB-D SLAMデータセットを
    PyTorch Dataset として提供し、各フレームを辞書形式のテンソルとして返します。

Returned keys (typical) / 返却キー例:
    - color: (H, W, 3) float32 in [0,1]
    - depth: (H, W) float32 meters
    - normal / error / angle / dx / dy: optional auxiliary depth cues
    - local: local neighborhood cue (dataset-dependent)
    - gt_depth / gt_c2w: optional ground-truth (if available)

Multi-sensor depth / 複数深度センサー:
    When multiple depth sources are enabled, keys are prefixed with
    "2_", "3_", ... (e.g., "2_depth", "3_normal", ...).
"""


import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from wimsoslam.common import as_intrinsics_matrix, normalization
from torch.utils.data import Dataset, Sampler

class SeqSampler(Sampler):
    """
    Sample a sequence of frames from a dataset.

    """
    def __init__(self, n_samples, step, include_last=True):
        self.n_samples = n_samples
        self.step = step
        self.include_last = include_last
    def __iter__(self):
        if self.include_last:
            return iter(list(range(0, self.n_samples, self.step)) + [self.n_samples - 1])
        else:
            return iter(range(0, self.n_samples, self.step))

    def __len__(self) -> int:
        return self.n_samples

def get_dataset(cfg, args, scale, device='cuda:0'):
    return dataset_dict[cfg['dataset']](cfg, args, scale, device=device)

def get_normal_map(depth_data, K, kernel_size=9, color_std=100, spatial_std=9):
    """
    Applies bilateral filter and cross product for normal information.

    """

    H, W = depth_data.shape

    # smooth depth while preserving edges
    depth_filtered = cv2.bilateralFilter(depth_data, kernel_size, color_std, spatial_std)

    # store vector differences
    f = np.zeros((3, H, W))
    t = np.zeros((3, H, W))

    # get pixel position vector
    x, y = np.meshgrid(np.arange(0, W), np.arange(0, H))
    x = x.reshape([-1])
    y = y.reshape([-1])
    xyz = np.vstack((x, y, np.ones_like(x)))

    # project from 2D to 3D
    pts_3d = np.dot(np.linalg.inv(K), xyz * depth_filtered.reshape([-1]))
    pts_3d_world = pts_3d.reshape((3, H, W))

    # get vector differences
    f[:, 1:-1, 1:-1] = pts_3d_world[:, 1 : H - 1, 2:W] - pts_3d_world[:, 1 : H - 1, 1 : W - 1]
    t[:, 1:-1, 1:-1] = pts_3d_world[:, 2:H, 1 : W - 1] - pts_3d_world[:, 1 : H - 1, 1 : W - 1]

    # get normal and normalize to unit vector
    normal_data = np.cross(f, t, axisa=0, axisb=0)
    normal_data = normalization(normal_data)

    return normal_data.astype(np.float32)

class BaseDataset(Dataset):
    def __init__(self, cfg, args, scale, device="cuda:0"):
        super(BaseDataset, self).__init__()
        self.name = cfg["dataset"]
        self.num_sensors = cfg["num_sensors"]
        self.device = device
        self.scale = scale
        self.png_depth_scale = cfg["cam"]["png_depth_scale"]

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = (
            cfg["cam"]["H"],
            cfg["cam"]["W"],
            cfg["cam"]["fx"],
            cfg["cam"]["fy"],
            cfg["cam"]["cx"],
            cfg["cam"]["cy"],
        )

        # resize the input images to crop_size (variable name used in lietorch)
        if "crop_size" in cfg["cam"]:
            crop_size = cfg["cam"]["crop_size"]
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx * self.fx
            self.fy = sy * self.fy
            self.cx = sx * self.cx
            self.cy = sy * self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here
        if cfg["cam"]["crop_edge"] > 0:
            self.H -= cfg["cam"]["crop_edge"] * 2
            self.W -= cfg["cam"]["crop_edge"] * 2
            self.cx -= cfg["cam"]["crop_edge"]
            self.cy -= cfg["cam"]["crop_edge"]

        self.distortion = np.array(cfg["cam"]["distortion"]) if "distortion" in cfg["cam"] else None
        self.crop_size = cfg["cam"]["crop_size"] if "crop_size" in cfg["cam"] else None  # TODO: maybe remove

        if args.input_folder is None:
            self.input_folder = cfg["data"]["input_folder"]
        else:
            self.input_folder = args.input_folder

        self.crop_edge = cfg["cam"]["crop_edge"]

    def __len__(self):
        return self.n_img

    def __getitem__(self, index):
        ret = {}
        prefix_keys = ["", "2_", "3_"]

        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        kern = 5
        kx, ky = cv2.getDerivKernels(0, 1, kern)
        norm_factor = (kx @ ky.T)[:, kern // 2 + 1 :].sum()

        edge = self.crop_edge

        # depth data
        depth_path = self.depth_paths[index]
        if ".png" in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        # print(depth_data)

        if self.name == "SevenScenes":
            depth_data[depth_data == 65535] = 0

        depth_data = depth_data.astype(np.float32) / self.png_depth_scale
        depth_data = depth_data * self.scale

        H, W = depth_data.shape

        # color data
        color_path = self.color_paths[index]
        color_data = cv2.imread(color_path)
        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color_data = cv2.undistort(color_data, K, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.0
        color_data = cv2.resize(color_data, (W, H))

        # GT depth data
        if self.gt_depth_paths:
            gt_depth_path = self.gt_depth_paths[index]
            if ".png" in gt_depth_path:
                gt_depth_data = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)
                gt_depth_data = gt_depth_data.astype(np.float32) / self.png_depth_scale
        else:
            gt_depth_data = depth_data

        # local ray data
        local_data = np.zeros((self.H, self.W, 3), np.float32)
        x, y = np.meshgrid(np.arange(0, self.W), np.arange(0, self.H))
        local_data[:, :, 0] = (x - K[0, 2]) / K[0, 0]
        local_data[:, :, 1] = (y - K[1, 2]) / K[1, 1]
        local_data[:, :, 2] = 1
        local_data = normalization(local_data)

        color_data = torch.from_numpy(color_data)
        gt_depth_data = torch.from_numpy(gt_depth_data)
        local_data = torch.from_numpy(local_data)

        # follow the pre-processing step in lietorch, actually is resize
        if self.crop_size is not None:
            color_data = color_data.permute(2, 0, 1)
            gt_depth_data = F.interpolate(gt_depth_data[None, None], self.crop_size, mode="nearest")[0, 0]
            color_data = F.interpolate(color_data[None], self.crop_size, mode="bilinear", align_corners=True)[0]
            color_data = color_data.permute(1, 2, 0).contiguous()

        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            gt_depth_data = gt_depth_data[edge:-edge, edge:-edge]

        for i in range(self.num_sensors):
            # depth data
            depth_path = self.depth_paths[i][index]
            if ".png" in depth_path:
                depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth_data = depth_data.astype(np.float32) / self.png_depth_scale
            depth_data = depth_data * self.scale

            depth_data = torch.from_numpy(depth_data)

            # follow the pre-processing step in lietorch, actually is resize
            if self.crop_size is not None:
                depth_data = F.interpolate(depth_data[None, None], self.crop_size, mode="nearest")[0, 0]

            # crop image edge, there are invalid value on the edge of the color image
            if edge > 0:
                depth_data = depth_data[edge:-edge, edge:-edge]

            # error data
            error_data = torch.abs(gt_depth_data - depth_data)
            error_data[depth_data == 0] = 0

            # normal data
            normal_data = get_normal_map(depth_data.numpy(), K)

            # angle data
            angle_data = np.sum(local_data.numpy() * normal_data, axis=2)

            # gradient data
            dx_data = cv2.Sobel(depth_data.numpy(), cv2.CV_32F, 1, 0, ksize=kern) / norm_factor
            dy_data = cv2.Sobel(depth_data.numpy(), cv2.CV_32F, 0, 1, ksize=kern) / norm_factor

            normal_data = torch.from_numpy(normal_data)
            angle_data = torch.from_numpy(angle_data)
            dx_data = torch.from_numpy(dx_data)
            dy_data = torch.from_numpy(dy_data)

            ret.update(
                {
                    f"{prefix_keys[i]}depth": depth_data.to(self.device),
                    f"{prefix_keys[i]}normal": normal_data.to(self.device),
                    f"{prefix_keys[i]}error": error_data.to(self.device),
                    f"{prefix_keys[i]}angle": angle_data.to(self.device),
                    f"{prefix_keys[i]}dx": dx_data.to(self.device),
                    f"{prefix_keys[i]}dy": dy_data.to(self.device),
                }
            )

        pose = self.poses[index]
        pose[:3, 3] *= self.scale

        ret.update(
            {
                "idx": index,
                "gt_c2w": pose.to(self.device),
                "color": color_data.to(self.device),
                "normal": normal_data.to(self.device),
                "local": local_data.to(self.device),
                "gt_depth": gt_depth_data.to(self.device),
            }
        )

        return ret, index, color_data, depth_data, pose

class Replica(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(Replica, self).__init__(cfg, args, scale, device)
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/results/frame*.jpg'))
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/results/depth*.png'))
        self.n_img = len(self.color_paths)
        self.load_poses(f'{self.input_folder}/traj.txt')
        self.gt_depth_paths = False

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)

class ScanNet(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(ScanNet, self).__init__(cfg, args, scale, device)
        # self.input_folder = os.path.join(self.input_folder, 'frames')
        self.color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.load_poses(os.path.join(self.input_folder, 'pose'))
        self.n_img = len(self.color_paths)

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)

class TUM_RGBD(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(TUM_RGBD, self).__init__(cfg, args, scale, device)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose


dataset_dict = {
    "replica": Replica,
    "scannet": ScanNet,
    "tumrgbd": TUM_RGBD
}