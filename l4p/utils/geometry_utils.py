# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import cv2
import einops
import torch
from typing import Tuple


def generate_point_map(depth_b1thw, intrinsics_b44t, world_T_cam_b44t):
    """Generate point map from depth, intrinsics and camera to world transformation

    Args:
        depth_b1thw (torch.Tensor): Depth map of shape (B, 1, T, H, W)
        intrinsics_b44t (torch.Tensor): Intrinsics of shape (B, 4, 4, T)
        world_T_cam_b44t (torch.Tensor): Camera to world transformation of shape (B, 4, 4, T)

    Returns:
        torch.Tensor: Point map of shape (B, 3, T, H, W)
    """
    device = depth_b1thw.device
    dtype = depth_b1thw.dtype
    B, _, T, H, W = depth_b1thw.shape

    point_map_b3thw = torch.zeros(B, 3, T, H, W, dtype=dtype, device=device)

    intrinsics_b33t = intrinsics_b44t[:, :3, :3]

    j, i = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    point_map_b3thw[:, 0, :, :, :] = i.expand(B, T, -1, -1)
    point_map_b3thw[:, 1, :, :, :] = j.expand(B, T, -1, -1)
    point_map_b3thw[:, 2, :, :, :] = 1

    point_map_b3thw = torch.einsum(
        "bmnt,bnthw->bmthw",
        torch.inverse(intrinsics_b33t.permute(0, 3, 1, 2).to(dtype=torch.float32)).permute(0, 2, 3, 1),
        point_map_b3thw,
    )

    # get point cloud
    point_map_b3thw = point_map_b3thw * depth_b1thw
    point_map_b4thw = torch.cat((point_map_b3thw, torch.ones_like(point_map_b3thw[:, :1])), dim=1)
    point_map_b4thw = torch.einsum("bmnt,bnthw->bmthw", world_T_cam_b44t, point_map_b4thw)
    point_map_b3thw = point_map_b4thw[:, :3].to(dtype=dtype)

    return point_map_b3thw


def unproject_2d_track_to_3d(
    track_xy_bn2t: torch.Tensor, track_Z_bn1t: torch.Tensor, intrinsics_b44t: torch.Tensor
) -> torch.Tensor:
    """Projects 2D track and depth into 3D in camera coordinate system.

    Args:
        track_xy_bn2t (torch.Tensor): 2d track with [x,y] positions
        track_Z_bn1t (torch.Tensor): depth of track
        intrinsics_b44t (torch.Tensor): camera intrinsics, assumes simple pinhole model [fx, fy, cx, cy]

    Returns:
        torch.Tensor: 3D tracking with [X,Y,Z] in camera coordinate system
    """
    track_X_bn1t = (
        (track_xy_bn2t[:, :, 0:1, :] - intrinsics_b44t[:, 0:1, 2:3, :])
        * track_Z_bn1t
        / intrinsics_b44t[:, 0:1, 0:1, :]
    )
    track_Y_bn1t = (
        (track_xy_bn2t[:, :, 1:2, :] - intrinsics_b44t[:, 1:2, 2:3, :])
        * track_Z_bn1t
        / intrinsics_b44t[:, 1:2, 1:2, :]
    )
    track_XYZ_bn3t = torch.cat([track_X_bn1t, track_Y_bn1t, track_Z_bn1t], dim=-2)

    return track_XYZ_bn3t


def generate_3d_track_point_map(
    track_2d_traj_bn2t: torch.Tensor,
    track_2d_depth_bn1t: torch.Tensor,
    intrinsics_b44t: torch.Tensor,
    world_T_cam_b44t: torch.Tensor,
) -> torch.Tensor:
    """
    Generates 3D track point map from 2D track, depth and camear information

    Args:
        track_2d_traj_bn2t (torch.Tensor): 2D track trajectory
        track_2d_depth_bn1t (torch.Tensor): 2D track depth
        intrinsics_b44t (torch.Tensor): Camera intrinsics
        world_T_cam_b44t (torch.Tensor): Camera to world transformation

    Returns:
        torch.Tensor: 3D track point map
    """
    XYZ_bn3t = unproject_2d_track_to_3d(track_2d_traj_bn2t, track_2d_depth_bn1t, intrinsics_b44t)
    XYZ_b3tn = einops.rearrange(XYZ_bn3t, "b n k t -> b k t n")
    XYZ_b4tn = torch.cat((XYZ_b3tn, torch.ones_like(XYZ_b3tn[:, :1])), dim=1)
    XYZ_b4tn = torch.einsum("bmnt,bntp->bmtp", world_T_cam_b44t, XYZ_b4tn)
    XYZ_bn3t = einops.rearrange(XYZ_b4tn[:, :3], "b k t n -> b n k t")
    return XYZ_bn3t


def normalize_intrinsics(intrinsics_b44t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Normalize intrinsics to be in the range [0, 1]"""
    intrinsics = torch.clone(intrinsics_b44t).detach()
    intrinsics[:, :2, 2] += 0.5
    intrinsics[:, 0] = intrinsics[:, 0] / w
    intrinsics[:, 1] = intrinsics[:, 1] / h
    return intrinsics


def denormalize_intrinsics(intrinsics_b44t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Denormalize intrinsics to be in the range [0, w] and [0, h]"""
    intrinsics = torch.clone(intrinsics_b44t).detach()
    intrinsics[:, 0] *= w
    intrinsics[:, 1] *= h
    intrinsics[:, :2, 2] -= 0.5
    return intrinsics


def get_cam_T_ref(cam_T_world_b44t: torch.Tensor, ref_idx: int = 0) -> torch.Tensor:
    """Convert camera poses to be relative to the first (reference) frame.

    Args:
        cam_T_world_b44t (torch.Tensor): Input transform in shape (B, 4, 4, T)
        ref_idx (int): Index of the reference frame

    Returns:
        torch.Tensor: Output transform relative to first frame in shape (B, T, 4, 4)
    """
    cam_T_world = cam_T_world_b44t.permute(0, 3, 1, 2)  # b x t x 4 x 4
    ref_T_world = cam_T_world[:, ref_idx : ref_idx + 1]  # b x 1 x 4 x 4
    world_T_ref = torch.linalg.inv(ref_T_world)
    cam_T_ref = torch.matmul(cam_T_world, world_T_ref)  # b x t x 4 x 4
    cam_T_ref_b44t = cam_T_ref.permute(0, 2, 3, 1)  # b x 4 x 4 x t
    return cam_T_ref_b44t


def scale_extrinsics(extrinsics_b44t: torch.Tensor, scale_b1: torch.Tensor) -> torch.Tensor:
    """Scale the translation part of extrinsics matrices while preserving rotation."""
    scaled_extrinsics = extrinsics_b44t.clone()
    scaled_extrinsics[:, :3, 3] = scaled_extrinsics[:, :3, 3] * scale_b1[:, None, None]
    return scaled_extrinsics


########################################################################################
# CAMERA TO PLUCKER RAY
########################################################################################


def scale_rays_plucker(camray_b6thw: torch.Tensor, scale_b1: torch.Tensor) -> torch.Tensor:
    """Scale the moments of the plucker coordinate system"""
    camray_scaled_b6thw = camray_b6thw.clone()
    camray_scaled_b6thw[:, 3:] = camray_scaled_b6thw[:, 3:] * scale_b1
    return camray_scaled_b6thw


def get_rays_plucker(
    intrinsics_b44t: torch.Tensor,
    extrinsics_b44t: torch.Tensor,
    emb_hw: Tuple[int, int],
    make_first_cam_ref: bool = True,
    normalize_dist: bool = False,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Get plucker coordinates for rays of an image

    Args:
        intrinsics_b44t (torch.Tensor): normalized intrinsics
        extrinsics_b44t (torch.Tensor): extrinsics
        emb_hw (Tuple[int, int]): (h, w) tuple containing embedding spatial size
        normalize_dist (bool): if true, normalize dist from 0th to 1st to unit length

    Returns:
        camray_b6thw (torch.Tensor): plucker coordinates of shape (B, 6, T, H, W)
        scale (torch.Tensor | None): scale applied to the poses of shape (B)

    Adapted from https://github.com/jasonyzhang/RayDiffusion/blob/main/ray_diffusion/utils/rays.py
    """
    device = intrinsics_b44t.device
    dtype = intrinsics_b44t.dtype
    B, _, _, T = intrinsics_b44t.shape
    h, w = emb_hw

    # transform to coordinate system of the reference (first frame)
    cam_T_world = extrinsics_b44t.permute(0, 3, 1, 2)  # b x t x 4 x 4
    world_T_cam = torch.linalg.inv(cam_T_world)
    if make_first_cam_ref:
        ref_T_world = cam_T_world[:, :1]  # b x 1 x 4 x 4
        ref_T_cam = torch.matmul(ref_T_world, world_T_cam)  # b x t x 4 x 4
    else:
        ref_T_cam = world_T_cam  # b x t x 4 x 4

    # scale extrinsics
    if normalize_dist:
        dist_cam1 = ref_T_cam[:, 1, :3, -1].norm(dim=1)  # b
        dist_cam1[dist_cam1 < eps] = 1.0
        scale = 1.0 / dist_cam1
    else:
        scale = None

    # scale intrinsics
    intrinsics_b33t = denormalize_intrinsics(intrinsics_b44t, h, w)[:, :3, :3]

    # Create pixel coordinates
    j, i = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    pixels = torch.stack(
        [i.expand(B, -1, -1), j.expand(B, -1, -1), torch.ones_like(i).expand(B, -1, -1)],
        dim=-1,
    )  # B, h, w, 3

    # Calculate ray directions
    rays_d = torch.einsum("btmn,bhwn->bthwm", torch.inverse(intrinsics_b33t.permute(0, 3, 1, 2)), pixels)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)  # b x T x h x w x 3

    # Transform ray directions to world space
    rays_d = torch.einsum("btmn,bthwn->bthwm", ref_T_cam[..., :3, :3], rays_d)  # b x T x h x w x 3

    # Get ray origins from camera pose
    rays_o = ref_T_cam[..., :3, 3]  # b x T x 3
    if normalize_dist:
        rays_o = rays_o * scale[:, None, None]

    # Compute Plucker coordinates
    rays_oxd = torch.cross(rays_o.reshape(B, T, 1, 1, 3), rays_d, dim=-1)  # b x T x h x w x 3
    plucker = torch.cat([rays_d, rays_oxd], dim=-1)  # b x T x h x w x 6

    camray_b6thw = plucker.permute(0, 4, 1, 2, 3)

    return camray_b6thw, scale


########################################################################################
# PLUCKER RAY TO CAMERA
########################################################################################


def intersect_skew_lines_high_dim(
    points: torch.Tensor, directions: torch.Tensor, mask: torch.Tensor | None = None
) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    """Find nearest point to the skew lines defined by points on the lines and their directions

    Args:
        points (torch.Tensor): Points on the lines, shape: B x R x 3
        directions (torch.Tensor): Directions of the lines, shape: B x R x 3
        mask (torch.Tensor | None): Mask for which lines to consider, shape: B x R

    Returns:
        p_intersect (torch.Tensor | None): Nearest point to the skew lines, shape: B x 3
        directions (torch.Tensor | None): Normalized directions of the lines, shape: B x R x 3

    Adapted from https://github.com/jasonyzhang/RayDiffusion/blob/main/ray_diffusion/utils/normalize.py
    Implements https://en.wikipedia.org/wiki/Line–line_intersection#Nearest_points_to_skew_lines
    """
    dim = points.shape[-1]

    if mask is None:
        mask = torch.ones_like(points[..., 0])
    directions = torch.nn.functional.normalize(directions, dim=-1)

    eye = torch.eye(dim, device=points.device, dtype=points.dtype)[None, None]
    I_min_cov = (eye - (directions[..., None] * directions[..., None, :])) * mask[..., None, None]
    sum_proj = I_min_cov.matmul(points[..., None]).sum(dim=-3)

    p_intersect = torch.linalg.lstsq(I_min_cov.float().sum(dim=-3), sum_proj.float()).solution[..., 0]
    p_intersect = p_intersect.to(points.dtype)

    if torch.any(torch.isnan(p_intersect)):
        print(p_intersect)
        return None, None
    return p_intersect, directions


def compute_optimal_rotation_alignment(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute optimal R that minimizes: || A - B @ R ||_F

    Args:
        A (torch.Tensor): rays in camera coord of shape (N, 3)
        B (torch.Tensor): rays in ref coord of shape (N, 3)

    Returns:
        R (torch.tensor): rotation matrix of shape (3, 3)

    Adapted from https://github.com/jasonyzhang/RayDiffusion/blob/main/ray_diffusion/utils/rays.py
    Implements https://en.wikipedia.org/wiki/Kabsch_algorithm via SVD decomposition
    """
    H = B.T @ A
    H = H.float()
    U, _, Vh = torch.linalg.svd(H, full_matrices=True)
    s = torch.linalg.det((U @ Vh).float())
    S_prime = torch.diag(torch.tensor([1, 1, torch.sign(s)], device=A.device))
    R = U @ S_prime @ Vh
    return R.T


def plucker_to_point_direction(
    camray_b6thw: torch.Tensor, normalize_moment: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert 6D plucker coordinates to point + direction representation <O, D>.

    Args:
        camray_b6thw (torch.Tensor): 6D plucker coordinates of shape (B, 6, T, H, W)
        normalize_moment (bool): if true, normalizes the moment

    Returns:
        points (torch.Tensor): points of shape (B, 3, T, H, W)
        direction (torch.Tensor): direction of shape (B, 3, T, H, W)
    """
    direction = camray_b6thw[:, :3]  # b3thw
    moment = camray_b6thw[:, 3:]  # b3thw
    if normalize_moment:
        c = torch.linalg.norm(direction, dim=1, keepdim=True)
        moment = moment / c
    points = torch.cross(direction, moment, dim=1)  # b3thw
    return points, direction


def rays_to_cameras(
    camray_b6thw: torch.Tensor,
    intrinsics_b44t: torch.Tensor,
    ctr_only: bool = False,
) -> Tuple[torch.Tensor | None, torch.Tensor]:
    """
    Function to convert plucker coordinates to camera extrinsics matrix
    Assumes input intrinsics are provided

    Args:
        camray_b6thw (torch.Tensor): 6D plucker coordinates of shape (B, 6, T, H, W)
        intrinsics_b44t (torch.Tensor): normalized intrinsics of shape (B, 4, 4, T)
        ctr_only (bool): do not calculate rotation if true

    Returns:
        extrinsics_b44t (torch.Tensor | None): extrinsics matrix of shape (B, 4, 4, T)
        camera_centers_bt3 (torch.Tensor): camera centers of shape (B, T, 3)

    Adapted from https://github.com/jasonyzhang/RayDiffusion/blob/main/ray_diffusion/utils/rays.py
    """
    device = intrinsics_b44t.device
    dtype = intrinsics_b44t.dtype

    if camray_b6thw.dtype != dtype:
        camray_b6thw = camray_b6thw.to(dtype)
    if camray_b6thw.device != device:
        camray_b6thw = camray_b6thw.to(device)

    B, _, T, h, w = camray_b6thw.shape

    # Find camera locations
    origins, directions = plucker_to_point_direction(camray_b6thw)
    origins_rs = origins.permute(0, 2, 3, 4, 1).reshape(-1, h * w, 3)  # (bt)(hw)3
    directions_rs = directions.permute(0, 2, 3, 4, 1).reshape(-1, h * w, 3)  # (bt)(hw)3
    camera_centers, _ = intersect_skew_lines_high_dim(origins_rs, directions_rs)  # (bt)3
    camera_centers_bt3 = camera_centers.reshape(B, T, 3)

    if ctr_only:
        return None, camera_centers_bt3

    # Scale intrinsics to cameray ray space
    intrinsics_b33t = denormalize_intrinsics(intrinsics_b44t, h, w)[:, :3, :3]

    # Create pixel coordinates
    j, i = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    pixels = torch.stack(
        [i.expand(B, -1, -1), j.expand(B, -1, -1), torch.ones_like(i).expand(B, -1, -1)],
        dim=-1,
    )  # b, h, w, 3

    # Calculate ray directions
    rays_d = torch.einsum("btmn,bhwn->bthwm", torch.inverse(intrinsics_b33t.permute(0, 3, 1, 2)), pixels)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)  # b x T x h x w x 3

    # Create extrinsics matrix and add rotation that optimally aligns rays
    extrinsics_b44t = torch.zeros_like(intrinsics_b44t)
    extrinsics_b44t[:, 3, 3] = 1.0
    for b in range(B):
        for t in range(T):
            # compute rotation that aligns ideal ray directions to estimated ray directions
            extrinsics_b44t[b, :3, :3, t] = compute_optimal_rotation_alignment(
                rays_d[b, t].reshape(-1, 3),
                directions[b, :, t].reshape(3, -1).T,  # (hw)3
            )

    # Construct and add translation to the extrinsics matrix
    translation_bt3 = -torch.matmul(
        extrinsics_b44t[:, :3, :3].permute(0, 3, 1, 2), camera_centers_bt3[..., None]
    ).squeeze(3)
    extrinsics_b44t[:, :3, -1] = translation_bt3.permute(0, 2, 1)

    return extrinsics_b44t, camera_centers_bt3


def compute_optimal_rotation_intrinsics(rays_origin, rays_target, z_threshold=1e-4, reproj_threshold=0.2):
    """
    Compute optimal rotation and intrinsics to align rays.
    Implements by finding the homography that aligns the rays.
    Homography is decomposed into rotation and intrinsic matrix using RQ decomposition.

    Args:
        rays_origin (torch.Tensor): source rays of shape (N, 3)
        rays_target (torch.Tensor): target rays of shape (N, 3)
        z_threshold (float): Threshold for z value to be considered valid.

    Returns:
        R (torch.tensor): rotation matrix of shape (3, 3)
        K (torch.tensor): intrinsic matrix of shape (3, 3)
        H (torch.tensor): homography matrix of shape (3, 3)

    Modified from https://github.com/jasonyzhang/RayDiffusion/blob/main/ray_diffusion/utils/rays.py
    """
    device = rays_origin.device
    z_mask = torch.logical_and(torch.abs(rays_target) > z_threshold, torch.abs(rays_origin) > z_threshold)[
        :, 2
    ]
    rays_target = rays_target[z_mask]
    rays_origin = rays_origin[z_mask]
    rays_origin = rays_origin[:, :2] / rays_origin[:, -1:]
    rays_target = rays_target[:, :2] / rays_target[:, -1:]

    A, _ = cv2.findHomography(
        rays_origin.cpu().numpy(),
        rays_target.cpu().numpy(),
        cv2.RANSAC,
        reproj_threshold,
    )
    A = torch.from_numpy(A).float().to(device)

    if torch.linalg.det(A) < 0:
        A = -A

    H = torch.linalg.inv(A.float())  # H = K @ R
    out = cv2.RQDecomp3x3(H.cpu().numpy())
    K = out[1]
    R = out[2]
    K = K / K[2, 2]

    K = torch.from_numpy(K).float().to(device)
    R = torch.from_numpy(R).float().to(device)

    return R, K, H


def rays_to_cameras_and_intrinsics(
    camray_b6thw: torch.Tensor,
    ctr_only: bool = False,
    reproj_threshold: float = 0.2,
    output_size: Tuple[int, int] = (16, 16),
    fixed_intrinsics: bool = False,
) -> Tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """
    Compute optimal rotation and intrinsics to align rays.
    Implements by finding the homography that aligns the rays.
    Homography is decomposed into rotation and intrinsic matrix using RQ decomposition.

    Args:
        camray_b6thw (torch.Tensor): 6D plucker coordinates of shape (B, 6, T, H, W)
        ctr_only (bool): do not calculate rotation if true
        reproj_threshold (float): threshold for reprojection error
        output_size (Tuple[int, int]): output size of the intrinsics
        fixed_intrinsics (bool): if true, use fixed intrinsics for all frames

    Returns:
        extrinsics_b44t (torch.Tensor | None): extrinsics matrix of shape (B, 4, 4, T)
        camera_centers_bt3 (torch.Tensor): camera centers of shape (B, T, 3)
        intrinsics_est_b44t (torch.Tensor): intrinsics matrix of shape (B, 4, 4, T)
    """
    if fixed_intrinsics:
        return rays_to_cameras_and_fixed_per_frame_intrinsics(
            camray_b6thw, ctr_only, reproj_threshold, output_size
        )
    else:
        return rays_to_cameras_and_variable_per_frame_intrinsics(
            camray_b6thw, ctr_only, reproj_threshold, output_size
        )


def rays_to_cameras_and_fixed_per_frame_intrinsics(
    camray_b6thw: torch.Tensor,
    ctr_only: bool = False,
    reproj_threshold: float = 0.2,
    output_size: Tuple[int, int] = (16, 16),
) -> Tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """
    Compute optimal rotation and fixed intrinsics to align rays.
    Modified from https://github.com/jasonyzhang/RayDiffusion/blob/main/ray_diffusion/utils/rays.py
    """

    device = camray_b6thw.device
    dtype = torch.float32

    B, _, T, h, w = camray_b6thw.shape

    # Find camera locations
    origins, directions = plucker_to_point_direction(camray_b6thw)
    origins_rs = origins.permute(0, 2, 3, 4, 1).reshape(-1, h * w, 3)  # (bt)(hw)3
    directions_rs = directions.permute(0, 2, 3, 4, 1).reshape(-1, h * w, 3)  # (bt)(hw)3
    camera_centers, _ = intersect_skew_lines_high_dim(origins_rs, directions_rs)  # (bt)3
    camera_centers_bt3 = camera_centers.reshape(B, T, 3)

    if ctr_only:
        return None, camera_centers_bt3

    # Identity intrinsics
    intrinsics_b33t = torch.eye(3).to(device=device).to(dtype=dtype)[None, :, :, None]
    intrinsics_b33t = intrinsics_b33t.repeat(B, 1, 1, T)

    # Create pixel coordinates
    j, i = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    pixels = torch.stack(
        [i.expand(B, -1, -1), j.expand(B, -1, -1), torch.ones_like(i).expand(B, -1, -1)],
        dim=-1,
    )  # b, h, w, 3

    # Compute optimal rotation to align rays
    extrinsics_b44t = torch.zeros(B, 4, 4, T, device=device, dtype=dtype)
    extrinsics_b44t[:, 3, 3] = 1.0
    intrinsics_est_b44t = torch.zeros_like(extrinsics_b44t)
    intrinsics_est_b44t[:, 3, 3] = 1.0
    intrinsics_est_b44t[:, 2, 2] = 1.0

    # Calculate ray directions with identity intrinsics
    rays_d = torch.einsum("btmn,bhwn->bthwm", torch.inverse(intrinsics_b33t.permute(0, 3, 1, 2)), pixels)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)  # b x T x h x w x 3

    # compute intrinsics from first frame only
    for b in range(B):
        t = 0
        _, K, _ = compute_optimal_rotation_intrinsics(
            rays_d[b, t].reshape(-1, 3).float(),  # (hw)3
            directions[b, :, t].reshape(3, -1).T.float(),  # (hw)3
            reproj_threshold=reproj_threshold,
        )
        intrinsics_est_b44t[b, :3, :3, :] = K[:, :, None].repeat(1, 1, T)

    # calculare ray directions with estimated intrinsics
    rays_d = torch.einsum(
        "btmn,bhwn->bthwm", torch.inverse(intrinsics_est_b44t[:, :3, :3].permute(0, 3, 1, 2)), pixels
    )
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)  # b x T x h x w x 3

    # compute rotations that align the ideal rays with the estimated rays
    for b in range(B):
        for t in range(T):
            extrinsics_b44t[b, :3, :3, t] = compute_optimal_rotation_alignment(
                rays_d[b, t].reshape(-1, 3),
                directions[b, :, t].reshape(3, -1).T,  # (hw)3
            )

    # Construct and add translation to the extrinsics matrix
    translation_bt3 = -torch.matmul(
        extrinsics_b44t[:, :3, :3].permute(0, 3, 1, 2), camera_centers_bt3[..., None]
    ).squeeze(3)
    extrinsics_b44t[:, :3, -1] = translation_bt3.permute(0, 2, 1)

    # change the intrinsics to the output size
    H, W = output_size
    intrinsics_est_b44t = denormalize_intrinsics(normalize_intrinsics(intrinsics_est_b44t, h, w), H, W)

    return extrinsics_b44t, camera_centers_bt3, intrinsics_est_b44t


def rays_to_cameras_and_variable_per_frame_intrinsics(
    camray_b6thw: torch.Tensor,
    ctr_only: bool = False,
    reproj_threshold: float = 0.2,
    output_size: Tuple[int, int] = (16, 16),
) -> Tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """
    Compute optimal rotation and fixed intrinsics to align rays.
    Modified from https://github.com/jasonyzhang/RayDiffusion/blob/main/ray_diffusion/utils/rays.py
    """

    device = camray_b6thw.device
    dtype = torch.float32  # camray_b6thw.dtype

    B, _, T, h, w = camray_b6thw.shape

    # Find camera locations
    origins, directions = plucker_to_point_direction(camray_b6thw)
    origins_rs = origins.permute(0, 2, 3, 4, 1).reshape(-1, h * w, 3)  # (bt)(hw)3
    directions_rs = directions.permute(0, 2, 3, 4, 1).reshape(-1, h * w, 3)  # (bt)(hw)3
    camera_centers, _ = intersect_skew_lines_high_dim(origins_rs, directions_rs)  # (bt)3
    camera_centers_bt3 = camera_centers.reshape(B, T, 3)

    if ctr_only:
        return None, camera_centers_bt3

    # Identity intrinsics
    intrinsics_b33t = torch.eye(3).to(device=device).to(dtype=dtype)[None, :, :, None]
    intrinsics_b33t = intrinsics_b33t.repeat(B, 1, 1, T)

    # Create pixel coordinates
    j, i = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    pixels = torch.stack(
        [i.expand(B, -1, -1), j.expand(B, -1, -1), torch.ones_like(i).expand(B, -1, -1)],
        dim=-1,
    )  # b, h, w, 3

    # Calculate ray directions
    rays_d = torch.einsum("btmn,bhwn->bthwm", torch.inverse(intrinsics_b33t.permute(0, 3, 1, 2)), pixels)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)  # b x T x h x w x 3

    # Compute optimal rotation to align rays
    extrinsics_b44t = torch.zeros(B, 4, 4, T, device=device, dtype=dtype)
    extrinsics_b44t[:, 3, 3] = 1.0
    intrinsics_est_b44t = torch.zeros_like(extrinsics_b44t)
    intrinsics_est_b44t[:, 3, 3] = 1.0
    intrinsics_est_b44t[:, 2, 2] = 1.0

    for b in range(B):
        for t in range(T):
            R, K, A = compute_optimal_rotation_intrinsics(
                rays_d[b, t].reshape(-1, 3).float(),  # (hw)3
                directions[b, :, t].reshape(3, -1).T.float(),  # (hw)3
                reproj_threshold=reproj_threshold,
            )
            extrinsics_b44t[b, :3, :3, t] = R
            intrinsics_est_b44t[b, :3, :3, t] = K

    # Construct and add translation to the extrinsics matrix
    translation_bt3 = -torch.matmul(
        extrinsics_b44t[:, :3, :3].permute(0, 3, 1, 2), camera_centers_bt3[..., None]
    ).squeeze(3)
    extrinsics_b44t[:, :3, -1] = translation_bt3.permute(0, 2, 1)

    # change the intrinsics
    H, W = output_size
    intrinsics_est_b44t = denormalize_intrinsics(normalize_intrinsics(intrinsics_est_b44t, h, w), H, W)

    return extrinsics_b44t, camera_centers_bt3, intrinsics_est_b44t
