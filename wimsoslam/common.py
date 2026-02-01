# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 ams-OSRAM AG
# Copyright 2026 Maolin Wang (modifications)
#
# This repository redistributes and modifies Apache-2.0 licensed components.
# Upstream attributions: see NOTICE.

"""wimsoslam.common

EN:
    Shared math utilities for WIMSO-SLAM.
    Includes camera pose conversions, ray sampling, patch extraction, and
    stratified / importance sampling helpers.

JP:
    WIMSO-SLAM の共通ユーティリティ。
    カメラ姿勢の変換、レイサンプリング、パッチ抽出、
    stratified / importance sampling などを提供します。
"""


import numpy as np
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]

    return K


def sample_pdf(bins, weights, N_samples, det=False, device='cuda:0'):
    """
    Hierarchical sampling in NeRF paper.
    """
    # Get pdf
    # weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    pdf = weights

    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


def random_select(l, k):
    """
    Random select k values from 0..l.

    """
    return list(np.random.permutation(np.array(range(l)))[:min(l, k)])

def get_rays_from_uv(i, j, c2ws, H, W, fx, fy, cx, cy, device):
    """
    Get corresponding rays from input uv.

    """
    if isinstance(c2ws, np.ndarray):
        c2ws = torch.from_numpy(c2ws).to(device)
    # dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1).to(device)
    # dirs = torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i, device=device)], -1).to(device)
    dirs = torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i, device=device)], -1)
    dirs = dirs.unsqueeze(-2)
    # dirs = dirs.reshape(-1, 1, 3)

    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # import os
    # os.mkdir(str(dirs.size()))
    # os.mkdir(str(c2ws.size()))
    if len(c2ws.size()) == 3:
        c2ws = c2ws.squeeze(0)
    rays_d = torch.sum(dirs * c2ws[None, :3, :3], -1)
    # rays_d = torch.sum(dirs * c2ws[:, None, :3, :3], -1)
    rays_o = c2ws[None, :3, -1].expand(rays_d.shape)
    # print(c2ws[None, :3, -1])
    # rays_o = c2ws[:, None, :3, -1].expand(rays_d.shape)
    
    # rays_d = torch.sum(dirs * c2ws[:3, :3], -1)
    # rays_o = c2ws[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d

def select_uv(i, j, n, frame_dict, feature_maps, device="cuda:0", kernel=3):
    """
    Select n uv from dense uv.

    """

    ray_dict = {}

    i = i.reshape(-1)
    j = j.reshape(-1)

    indices = torch.randint(i.shape[0], (n,), device=device)
    indices = indices.clamp(0, i.shape[0])

    i = i[indices]  # (n)
    j = j[indices]  # (n)

    for feature in feature_maps:
        if frame_dict[feature].shape[-1] == 3:
            ray_dict[feature] = frame_dict[feature].reshape(-1, 3)
        else:
            ray_dict[feature] = frame_dict[feature].reshape(-1)
        ray_dict[feature] = ray_dict[feature][indices]

        for n in range(kernel):
            for m in range(kernel):
                mlp_index = n * kernel + m
                temp_feature_map = torch.roll(
                    frame_dict[feature],
                    shifts=(n - kernel // 2, m - kernel // 2),
                    dims=(0, 1),
                )
                if frame_dict[feature].shape[-1] == 3:
                    ray_dict[f"{feature}_{mlp_index}"] = temp_feature_map.reshape(-1, 3)
                else:
                    ray_dict[f"{feature}_{mlp_index}"] = temp_feature_map.reshape(-1)
                ray_dict[f"{feature}_{mlp_index}"] = ray_dict[f"{feature}_{mlp_index}"][indices]  # (n)

    # print(feature_maps,n,m)
    return i, j, ray_dict


def get_sample_uv(H0, H1, W0, W1, n, frame_dict, feature_maps, device="cuda:0", kernel=3):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1
    """

    for feature in feature_maps:
        # print(feature)
        # print(H0,H1, W0,W1)
        # try:
        #     aa = frame_dict[feature].size()
        #     # print(type(frame_dict))
        # except:
        #     print(feature_maps)
        #     print(feature)
        #     print(type(frame_dict))
        #     print(frame_dict.size())
        #     print(torch.tensor(frame_dict[feature]).size())

        # if isinstance(frame_dict, torch.Tensor):
        #     print(frame_dict.size())
        #     temp = frame_dict[H0:H1, W0:W1]
        #     frame_dict = {}
        #     frame_dict[feature] = temp
        #     break
        frame_dict[feature] = frame_dict[feature][H0:H1, W0:W1]
    i, j = torch.meshgrid(
        torch.linspace(W0, W1 - 1, W1 - W0).to(device),
        torch.linspace(H0, H1 - 1, H1 - H0).to(device),
    )
    i = i.t()  # transpose
    j = j.t()
    i, j, ray_dict = select_uv(i, j, n, frame_dict, feature_maps, device=device, kernel=kernel)
    return i, j, ray_dict


def get_samples(H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, frame_dict, feature_maps, device, kernel=3):
    """
    Get n rays from the image region H0..H1, W0..W1.
    c2w is its camera pose and depth/color is the corresponding image tensor.
    feature_maps: ['depth', 'norm', 'color', 'error']
    """

    i, j, ray_dict = get_sample_uv(H0, H1, W0, W1, n, frame_dict, feature_maps, device=device, kernel=kernel)
    rays_o, rays_d = get_rays_from_uv(i, j, frame_dict["est_c2w"], H, W, fx, fy, cx, cy, device)
    return rays_o, rays_d, ray_dict

def matrix_to_cam_pose(batch_matrices, RT=True):
    """
    Convert transformation matrix to quaternion and translation.
    Args:
        batch_matrices: (B, 4, 4)
        RT: if True, return (B, 7) with [R, T], else return (B, 7) with [T, R]
    Returns:
        (B, 7) with [R, T] or [T, R]
    """
    if RT:
        return torch.cat([matrix_to_quaternion(batch_matrices[:,:3,:3]), batch_matrices[:,:3,3]], dim=-1)
    else:
        return torch.cat([batch_matrices[:, :3, 3], matrix_to_quaternion(batch_matrices[:, :3, :3])], dim=-1)

def cam_pose_to_matrix(batch_poses):
    """
    Convert quaternion and translation to transformation matrix.
    Args:
        batch_poses: (B, 7) with [R, T] or [T, R]
    Returns:
        (B, 4, 4) transformation matrix
    """
    c2w = torch.eye(4, device=batch_poses.device).unsqueeze(0).repeat(batch_poses.shape[0], 1, 1)
    c2w[:,:3,:3] = quaternion_to_matrix(batch_poses[:,:4])
    c2w[:,:3,3] = batch_poses[:,4:]

    return c2w

def get_rays(H, W, fx, fy, cx, cy, c2w, device):
    """
    Get rays for a whole image.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()  # transpose
    j = j.t()
    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(H, W, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def normalize_3d_coordinate(p, bound):
    """
    Normalize 3d coordinate to [-1, 1] range.
    Args:
        p: (N, 3) 3d coordinate
        bound: (3, 2) min and max of each dimension
    Returns:
        (N, 3) normalized 3d coordinate

    """
    p = p.reshape(-1, 3)
    p[:, 0] = ((p[:, 0]-bound[0, 0])/(bound[0, 1]-bound[0, 0]))*2-1.0
    p[:, 1] = ((p[:, 1]-bound[1, 0])/(bound[1, 1]-bound[1, 0]))*2-1.0
    p[:, 2] = ((p[:, 2]-bound[2, 0])/(bound[2, 1]-bound[2, 0]))*2-1.0
    return p

def normalization(data):
    """Normalize 3D vector direction."""
    mo_chang = np.sqrt(
        np.multiply(data[:, :, 0], data[:, :, 0])
        + np.multiply(data[:, :, 1], data[:, :, 1])
        + np.multiply(data[:, :, 2], data[:, :, 2])
    )
    mo_chang = np.dstack((mo_chang, mo_chang, mo_chang))
    return data / (mo_chang + 1e-9)
