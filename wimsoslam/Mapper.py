# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 ams-OSRAM AG
# Copyright 2026 Maolin Wang (modifications)
#
# This repository redistributes and modifies Apache-2.0 licensed components.
# Upstream attributions: see NOTICE.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import time

from colorama import Fore, Style

from wimsoslam.common import (get_samples, random_select, matrix_to_cam_pose, cam_pose_to_matrix)
from wimsoslam.utils.datasets import get_dataset, SeqSampler
from wimsoslam.utils.Frame_Visualizer import Frame_Visualizer
from wimsoslam.tools.cull_mesh import cull_mesh

class Mapper(object):
    """
    Mapping main class.
    Args:
        cfg (dict): config dict
        args (argparse.Namespace): arguments
        slam_sys (WIMSO-SLAM): WIMSO-SLAM object
    """

    def __init__(self, cfg, args, slam_sys):

        self.cfg = cfg
        self.args = args
        self.prefix_keys = ["", "2_", "3_"]

        # control the features used in mapping process TODO: move to config-driven feature selection
        self.feature_maps = ["color", "local"]
        # self.feature_maps = ["color", "local", "est_c2w"]
        depth_features = ["depth", "normal", "error", "angle", "dx", "dy"]
        for i in range(self.cfg["num_sensors"]):
            prefix = self.prefix_keys[i]
            self.feature_maps += [f"{prefix}{feature}" for feature in depth_features]

        self.idx = slam_sys.idx
        self.truncation = slam_sys.truncation
        self.bound = slam_sys.bound
        self.logger = slam_sys.logger
        self.mesher = slam_sys.mesher
        self.output = slam_sys.output
        self.verbose = slam_sys.verbose
        self.renderer = slam_sys.renderer
        self.mapping_idx = slam_sys.mapping_idx
        self.mapping_cnt = slam_sys.mapping_cnt
        self.decoders = slam_sys.shared_decoders

        self.planes_xy = slam_sys.shared_planes_xy
        self.planes_xz = slam_sys.shared_planes_xz
        self.planes_yz = slam_sys.shared_planes_yz

        self.c_planes_xy = slam_sys.shared_c_planes_xy
        self.c_planes_xz = slam_sys.shared_c_planes_xz
        self.c_planes_yz = slam_sys.shared_c_planes_yz

        self.estimate_c2w_list = slam_sys.estimate_c2w_list
        self.mapping_first_frame = slam_sys.mapping_first_frame

        self.scale = cfg['scale']
        self.device = cfg['device']
        self.keyframe_device = cfg['keyframe_device']

        self.eval_rec = cfg['meshing']['eval_rec']
        self.joint_opt = False  # Even if joint_opt is enabled, it starts only when there are at least 4 keyframes
        self.joint_opt_cam_lr = cfg['mapping']['joint_opt_cam_lr'] # The learning rate for camera poses during mapping
        self.mesh_freq = cfg['mapping']['mesh_freq']
        self.ckpt_freq = cfg['mapping']['ckpt_freq']
        self.mapping_pixels = cfg['mapping']['pixels']
        self.every_frame = cfg['mapping']['every_frame']
        self.w_sdf_fs = cfg['mapping']['w_sdf_fs']
        self.w_sdf_center = cfg['mapping']['w_sdf_center']
        self.w_sdf_tail = cfg['mapping']['w_sdf_tail']
        self.w_depth = cfg['mapping']['w_depth']
        self.w_color = cfg['mapping']['w_color']
        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.mapping_window_size = cfg['mapping']['mapping_window_size']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']
        self.no_log_on_first_frame = cfg['mapping']['no_log_on_first_frame']
        self.no_mesh_on_first_frame = cfg['mapping']['no_mesh_on_first_frame']
        self.keyframe_selection_method = cfg['mapping']['keyframe_selection_method']

        self.keyframe_dict = []
        self.keyframe_list = []
        self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, num_workers=1, pin_memory=False,
                                       prefetch_factor=2, sampler=SeqSampler(self.n_img, self.every_frame))

        self.visualizer = Frame_Visualizer(freq=cfg['mapping']['vis_freq'], inside_freq=cfg['mapping']['vis_inside_freq'],
                                           vis_dir=os.path.join(self.output, 'mapping_vis'), renderer=self.renderer,
                                           truncation=self.truncation, verbose=self.verbose, device=self.device)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam_sys.H, slam_sys.W, slam_sys.fx, slam_sys.fy, slam_sys.cx, slam_sys.cy

    def sdf_losses(self, sdf, z_vals, gt_depth):
        """
        Computes the losses for a signed distance function (SDF) given its values, depth values and ground truth depth.

        Args:
        - self: instance of the class containing this method
        - sdf: a tensor of shape (R, N) representing the SDF values
        - z_vals: a tensor of shape (R, N) representing the depth values
        - gt_depth: a tensor of shape (R,) containing the ground truth depth values

        Returns:
        - sdf_losses: a scalar tensor representing the weighted sum of the free space, center, and tail losses of SDF
        """

        front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation),
                                 torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation),
                                torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * self.truncation)) *
                                  (z_vals < (gt_depth[:, None] + 0.4 * self.truncation)),
                                  torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

        fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
        center_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
        tail_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))

        sdf_losses = self.w_sdf_fs * fs_loss + self.w_sdf_center * center_loss + self.w_sdf_tail * tail_loss

        return sdf_losses

    def keyframe_selection_overlap(self, frame_dict, num_keyframes, num_samples=8, num_rays=50):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            frame_dict: ground truth color image of the current frame.
            num_keyframes (int): number of overlapping keyframes to select.
            num_samples (int, optional): number of samples/points per ray. Defaults to 8.
            num_rays (int, optional): number of pixels to sparsely sample
                from each image. Defaults to 50.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        rays_o, rays_d, ray_dict = get_samples(
            0, H, 0, W, num_rays, H, W, fx, fy, cx, cy, frame_dict, self.feature_maps, self.device)

        gt_depth = ray_dict["depth"].reshape(-1, 1)
        nonzero_depth = gt_depth[:, 0] > 0
        rays_o = rays_o[nonzero_depth]
        rays_d = rays_d[nonzero_depth]
        gt_depth = gt_depth[nonzero_depth]
        gt_depth = gt_depth.repeat(1, num_samples)
        t_vals = torch.linspace(0., 1., steps=num_samples).to(device)
        near = gt_depth*0.8
        far = gt_depth+0.5
        z_vals = near * (1.-t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [num_rays, num_samples, 3]
        pts = pts.reshape(1, -1, 3)

        keyframes_c2ws = torch.stack([self.estimate_c2w_list[idx] for idx in self.keyframe_list], dim=0)
        # print(keyframes_c2ws)
        w2cs = torch.inverse(keyframes_c2ws[:-2])     ## The last two keyframes are already included

        ones = torch.ones_like(pts[..., 0], device=device).reshape(1, -1, 1)
        homo_pts = torch.cat([pts, ones], dim=-1).reshape(1, -1, 4, 1).expand(w2cs.shape[0], -1, -1, -1)
        w2cs_exp = w2cs.unsqueeze(1).expand(-1, homo_pts.shape[1], -1, -1)
        cam_cords_homo = w2cs_exp @ homo_pts
        cam_cords = cam_cords_homo[:, :, :3]
        K = torch.tensor([[fx, .0, cx], [.0, fy, cy],
                          [.0, .0, 1.0]], device=device).reshape(3, 3)
        cam_cords[:, :, 0] *= -1
        uv = K @ cam_cords
        z = uv[:, :, -1:] + 1e-5
        uv = uv[:, :, :2] / z
        edge = 20
        mask = (uv[:, :, 0] < W - edge) * (uv[:, :, 0] > edge) * \
               (uv[:, :, 1] < H - edge) * (uv[:, :, 1] > edge)
        mask = mask & (z[:, :, 0] < 0)
        mask = mask.squeeze(-1)
        percent_inside = mask.sum(dim=1) / uv.shape[1]

        ## Considering only overlapped frames
        selected_keyframes = torch.nonzero(percent_inside).squeeze(-1)
        rnd_inds = torch.randperm(selected_keyframes.shape[0])
        selected_keyframes = selected_keyframes[rnd_inds[:num_keyframes]]

        selected_keyframes = list(selected_keyframes.cpu().numpy())

        return selected_keyframes

    def optimize_mapping(self, iters, lr_factor, idx, frame_dict, keyframe_dict, keyframe_list, cur_c2w):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if joint_opt enables).

        Args:
            iters (int): number of mapping iterations.
            lr_factor (float): the factor to times on current lr.
            idx (int): the index of current frame
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (tensor): gt_depth image of the current camera.
            gt_cur_c2w (tensor): groundtruth camera to world matrix corresponding to current frame.
            frame_dict (list) : a list of dictionaries of frames info.
            keyframe_dict (list): a list of dictionaries of keyframes info.
            keyframe_list (list): list of keyframes indices.
            cur_c2w (tensor): the estimated camera to world matrix of current frame. 

        Returns:
            cur_c2w: return the updated cur_c2w, return the same input cur_c2w if no joint_opt
        """
        cur_gt_color, cur_gt_depth, gt_cur_c2w = frame_dict["color"], frame_dict["depth"], frame_dict["gt_c2w"]
        all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        cfg = self.cfg
        device = self.device

        if len(keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == 'global':
                optimize_frame = random_select(len(self.keyframe_dict)-2, self.mapping_window_size-1)
            elif self.keyframe_selection_method == 'overlap':
                # optimize_frame = self.keyframe_selection_overlap(cur_gt_color, cur_gt_depth, cur_c2w, self.mapping_window_size-1)
                optimize_frame = self.keyframe_selection_overlap(frame_dict, self.mapping_window_size-1)

        # add the last two keyframes and the current frame(use -1 to denote)
        if len(keyframe_list) > 1:
            optimize_frame = optimize_frame + [len(keyframe_list)-1] + [len(keyframe_list)-2]
            optimize_frame = sorted(optimize_frame)
        optimize_frame += [-1]  ## -1 represents the current frame

        pixs_per_image = self.mapping_pixels//len(optimize_frame)

        decoders_para_list = []
        # decoders_para_list += list(self.decoders.parameters())
        decoders_para_list += list(self.decoders.linears.parameters())
        decoders_para_list += list(self.decoders.c_linears.parameters())
        decoders_para_list += list(self.decoders.output_linear.parameters())
        decoders_para_list += list(self.decoders.c_output_linear.parameters())
        # decoders_para_list += list(self.decoders.fine_decoder.parameters())

        planes_para = []
        for planes in [self.planes_xy, self.planes_xz, self.planes_yz]:
            for i, plane in enumerate(planes):
                plane = nn.Parameter(plane)
                planes_para.append(plane)
                planes[i] = plane

        c_planes_para = []
        for c_planes in [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]:
            for i, c_plane in enumerate(c_planes):
                c_plane = nn.Parameter(c_plane)
                c_planes_para.append(c_plane)
                c_planes[i] = c_plane

        gt_depths = []
        gt_colors = []
        c2ws = []
        gt_c2ws = []
        optim_frame_dict = {}

        for frame in optimize_frame:
            # the oldest frame should be fixed to avoid drifting
            if frame != -1:
                # print(keyframe_dict)
                # print(self.feature_maps)
                # if isinstance(keyframe_dict,list):
                #     keyframe_dict = keyframe_dict[0]
                # print(keyframe_dict[frame].keys())
                for feature in self.feature_maps:
                    optim_frame_dict[feature] = keyframe_dict[frame][feature].to(device)
                optim_frame_dict["est_c2w"] = torch.tensor(keyframe_dict[frame]["est_c2w"]).to(device)
                gt_depths.append(keyframe_dict[frame]['depth'].to(device))
                gt_colors.append(keyframe_dict[frame]['color'].to(device))
                # c2ws.append(keyframe_dict[frame]['est_c2w'])
                gt_c2ws.append(keyframe_dict[frame]['gt_c2w'])
                # print(keyframe_dict[frame]['est_c2w'])
            else:
                for feature in self.feature_maps:
                    optim_frame_dict[feature] = frame_dict[feature].to(device)
                optim_frame_dict["est_c2w"] = frame_dict["est_c2w"].to(device)
                gt_depths.append(cur_gt_depth)
                gt_colors.append(cur_gt_color)
                c2ws.append(cur_c2w)
                gt_c2ws.append(gt_cur_c2w)
                # print(cur_c2w)
        gt_depths = torch.stack(gt_depths, dim=0)
        gt_colors = torch.stack(gt_colors, dim=0)
        # if isinstance(c2ws,list):
        #     c2ws = torch.stack(c2ws, dim=0)
        # else:
        #     c2ws = torch.stack(c2ws, dim=0)
        c2ws = torch.stack(c2ws, dim=0)
        # print(optim_frame_dict["est_c2w"])
        # tensor([[-3.2057e-01, -4.4806e-01,  8.3455e-01,  3.4530e+00],
        # [ 9.4722e-01, -1.5164e-01,  2.8244e-01,  4.5461e-01],
        # [ 1.0790e-16,  8.8105e-01,  4.7302e-01,  5.9363e-01],
        # [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]], device='cuda:0')

        uncertainty_params = []
        for decoder in self.decoders.conf_decoders:
            uncertainty_params += list(decoder.parameters())
        uncertainty_params += list(self.decoders.color_conf_decoder.parameters())


        if self.joint_opt:
            cam_poses = nn.Parameter(matrix_to_cam_pose(c2ws[1:]))

            optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                          {'params': planes_para, 'lr': 0},
                                          {'params': c_planes_para, 'lr': 0},
                                          {"params": uncertainty_params, "lr": 0},
                                          {'params': [cam_poses], 'lr': 0}])

        else:
            optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                          {'params': planes_para, 'lr': 0},
                                          {'params': c_planes_para, 'lr': 0},
                                          {"params": uncertainty_params, "lr": 0}])

        optimizer.param_groups[0]['lr'] = cfg['mapping']['lr']['decoders_lr'] * lr_factor
        optimizer.param_groups[1]['lr'] = cfg['mapping']['lr']['planes_lr'] * lr_factor
        optimizer.param_groups[2]['lr'] = cfg['mapping']['lr']['c_planes_lr'] * lr_factor
        optimizer.param_groups[3]['lr'] = cfg['mapping']['lr']['var_lr'] * lr_factor

        if self.joint_opt:
            optimizer.param_groups[4]['lr'] = self.joint_opt_cam_lr

        for joint_iter in range(iters):
            if (not (idx == 0 and self.no_vis_on_first_frame)):
                self.visualizer.save_imgs(idx, joint_iter, frame_dict, self.feature_maps, cur_gt_depth, cur_gt_color, cur_c2w, all_planes, self.decoders)

            if self.joint_opt:
                ## We fix the oldest c2w to avoid drifting
                c2ws_ = torch.cat([c2ws[0:1], cam_pose_to_matrix(cam_poses)], dim=0)
            else:
                c2ws_ = c2ws

            batch_rays_o, batch_rays_d, batch_dict = get_samples(
                0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, optim_frame_dict, self.feature_maps, device, kernel=self.cfg["patch_size"])
            # batch_gt_depth = batch_dict["depth"]
            # batch_gt_color = batch_dict["color"]
            batch_dict["rays_o"] = batch_rays_o.float()
            batch_dict["rays_d"] = batch_rays_d.float()
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0).to(device) - det_rays_o) / det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_dict["depth"]
            print('Mapper:',sum(inside_mask))
                
            for feature in self.feature_maps:
                batch_dict[feature] = batch_dict[feature][inside_mask]
                # print(feature,batch_dict[feature].size())
                for mlp_index in range(self.cfg["patch_size"] ** 2):
                    batch_dict[f"{feature}_{mlp_index}"] = batch_dict[f"{feature}_{mlp_index}"][inside_mask]
            for feature in ["rays_o", "rays_d"]:
                batch_dict[feature] = batch_dict[feature][inside_mask]
            
            # for feature in batch_dict.keys():
            #     print(feature,batch_dict[feature].size())
            # batch_rays_d = batch_rays_d[inside_mask]
            # batch_rays_o = batch_rays_o[inside_mask]
            # batch_gt_depth = batch_gt_depth[inside_mask]
            # batch_gt_color = batch_gt_color[inside_mask]
            # print(batch_rays_d.size())
            # print(batch_rays_o.size())
            # print(batch_gt_depth.size())
            # print(batch_gt_color.size())
            # print(inside_mask.size())
            

            # depth, color, sdf, z_vals, var, uncertainty = self.renderer.render_batch_ray(all_planes, self.decoders, batch_rays_d,
            #                                                            batch_rays_o, batch_dict, device, self.truncation,
            #                                                            gt_depth=batch_gt_depth)
            depth, color, sdf, z_vals, var, uncertainty = self.renderer.render_batch_ray(all_planes, self.decoders, batch_dict["rays_d"], batch_dict["rays_o"],
                                                                   batch_dict, self.device, self.truncation, gt_depth=batch_dict['depth'])
            # depth_mask = (batch_dict["depth"] > 0)

            for i in range(self.cfg["num_sensors"]):
                prefix = self.prefix_keys[i]

                depth_mask = batch_dict[f"{prefix}depth"] > 0

                ## SDF losses
                # print(sdf.size())
                # print(z_vals.size())
                # print(batch_dict[f"{prefix}depth"].size())
                # print(depth_mask.size())
                loss = self.sdf_losses(sdf[depth_mask], z_vals[depth_mask], batch_dict[f"{prefix}depth"][depth_mask])

                ## Color Loss
                # print(batch_dict["color"].size())
                # print(color.size())
                # print(var[..., -1].size())
                # torch.Size([4000, 3])
                # torch.Size([4000, 3])
                # torch.Size([4000])
                maxtr = var[..., -1].repeat(3,1).T
                loss = loss + self.w_color * (torch.square(batch_dict["color"] - color) / (1e-3 + maxtr))[depth_mask].mean()

                ### Depth loss
                loss = loss + self.w_depth * (
                            (torch.square(batch_dict[f"{prefix}depth"][depth_mask] - depth[depth_mask])) / (
                            torch.sqrt(uncertainty + 1e-10) + (var[:, i]))).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                return loss.item()

        if self.joint_opt:
            # put the updated camera poses back
            optimized_c2ws = cam_pose_to_matrix(cam_poses.detach())

            camera_tensor_id = 0
            for frame in optimize_frame[1:]:
                if frame != -1:
                    keyframe_dict[frame]['est_c2w'] = optimized_c2ws[camera_tensor_id]
                    camera_tensor_id += 1
                else:
                    cur_c2w = optimized_c2ws[-1]

        return cur_c2w

    def run(self):
        def run(self):
            """
            Runs the mapping thread for the input RGB-D frames.

            Args:
                None

            Returns:
                None
        """
        cfg = self.cfg
        all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)
        frame_dict, _, _, _, _ = self.frame_reader[0]
        idx = frame_dict["idx"]
        gt_color = frame_dict["color"]
        gt_depth = frame_dict["depth"]
        gt_local = frame_dict["local"]
        gt_c2w = frame_dict["gt_c2w"]
        gt_normal = frame_dict["normal"]
        gt_error = frame_dict["error"]
        gt_angle = frame_dict["angle"]
        gt_dx = frame_dict["dx"]
        gt_dy = frame_dict["dy"]

        data_iterator = iter(self.frame_loader)

        ## Fixing the first camera pose
        self.estimate_c2w_list[0] = gt_c2w

        init_phase = True
        prev_idx = -1
        while True:
            while True:
                idx = self.idx[0].clone()
                if idx == self.n_img-1: ## Last input frame
                    break

                if idx % self.every_frame == 0 and idx != prev_idx:
                    break

                time.sleep(0.001)

            prev_idx = idx

            if self.verbose:
                print(Fore.GREEN)
                print("Mapping Frame ", idx.item())
                print(Style.RESET_ALL)

            _, _, gt_color, gt_depth, gt_c2w = next(data_iterator)
            gt_color = gt_color.squeeze(0).to(self.device, non_blocking=True)
            gt_depth = gt_depth.squeeze(0).to(self.device, non_blocking=True)
            gt_c2w = gt_c2w.squeeze(0).to(self.device, non_blocking=True)

            cur_c2w = self.estimate_c2w_list[idx]
            frame_dict["est_c2w"] = cur_c2w

            # copy images into modifiable dictionary
            optim_frame_dict = {}

            for key in frame_dict:
                optim_frame_dict[key] = frame_dict[key]
                # optim_frame_dict[key] = frame_dict[key].clone()

            if not init_phase:
                lr_factor = cfg['mapping']['lr_factor']
                iters = cfg['mapping']['iters']
            else:
                lr_factor = cfg['mapping']['lr_first_factor']
                iters = cfg['mapping']['iters_first']

            ## Deciding if camera poses should be jointly optimized
            self.joint_opt = (len(self.keyframe_list) > 4) and cfg['mapping']['joint_opt']

            # cur_c2w = self.optimize_mapping(iters, lr_factor, idx, optim_frame_dict,
            #                                 self.keyframe_dict, self.keyframe_list)
            cur_c2w = self.optimize_mapping(iters, lr_factor, idx, optim_frame_dict,
                                            self.keyframe_dict, self.keyframe_list, cur_c2w)
            # cur_c2w = self.optimize_mapping(iters, lr_factor, idx, gt_color, gt_depth, gt_c2w,
            #                                 self.keyframe_dict, self.keyframe_list, cur_c2w)

            if self.joint_opt:
                self.estimate_c2w_list[idx] = cur_c2w

            # add new frame to keyframe set
            if idx % self.keyframe_every == 0:
                self.keyframe_list.append(idx)
                # self.keyframe_dict.append({'gt_c2w': gt_c2w, 'idx': idx, 'color': gt_color.to(self.keyframe_device),
                #                            'depth': gt_depth.to(self.keyframe_device), 'est_c2w': cur_c2w.clone()})
                self.keyframe_dict.append({'gt_c2w': gt_c2w, 'idx': idx, 'color': gt_color.to(self.keyframe_device), 'local': gt_local.to(self.keyframe_device),
                                           'normal': gt_normal.to(self.keyframe_device),'dy': gt_dy.to(self.keyframe_device),'dx': gt_dx.to(self.keyframe_device),'angle': gt_angle.to(self.keyframe_device),'error': gt_error.to(self.keyframe_device),'depth': gt_depth.to(self.keyframe_device), 'est_c2w': cur_c2w})

            init_phase = False
            self.mapping_first_frame[0] = 1     # mapping of first frame is done, can begin tracking

            if ((not (idx == 0 and self.no_log_on_first_frame)) and idx % self.ckpt_freq == 0) or idx == self.n_img-1:
                self.logger.log(idx, self.keyframe_list)

            self.mapping_idx[0] = idx
            self.mapping_cnt[0] += 1

            if (idx % self.mesh_freq == 0) and (not (idx == 0 and self.no_mesh_on_first_frame)):
                mesh_out_file = f'{self.output}/mesh/{idx:05d}_mesh.ply'
                self.mesher.get_mesh(mesh_out_file, all_planes, self.decoders, self.keyframe_dict, self.device)
                cull_mesh(mesh_out_file, self.cfg, self.args, self.device, estimate_c2w_list=self.estimate_c2w_list[:idx+1])

            if idx == self.n_img-1:
                if self.eval_rec:
                    mesh_out_file = f'{self.output}/mesh/final_mesh_eval_rec.ply'
                else:
                    mesh_out_file = f'{self.output}/mesh/final_mesh.ply'

                self.mesher.get_mesh(mesh_out_file, all_planes, self.decoders, self.keyframe_dict, self.device)
                cull_mesh(mesh_out_file, self.cfg, self.args, self.device, estimate_c2w_list=self.estimate_c2w_list)

                break

            if idx == self.n_img-1:
                break