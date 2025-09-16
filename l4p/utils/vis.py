# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import numpy as np
import mediapy as media
import cv2
import open3d as o3d
from typing import List
import torch
import einops
from l4p.utils.misc import apply_fn
from l4p.utils.geometry_utils import (
    normalize_intrinsics,
    rays_to_cameras,
    get_cam_T_ref,
    generate_3d_track_point_map,
    generate_point_map,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as pltcm


##############################################################################################
# WRAPPER FUNCTIONS FOR VISUALIZING THE DATA
##############################################################################################
def generate_video_visualizations(batch, out, tasks, out_path=None):
    """Generate video visualizations for a batch of data.

    Args:
        batch (Dict[str, Any]): Batch of data to process and visualize.
        out (Dict[str, Any]): Output of the model.
        tasks (List[str]): Tasks to visualize.
        out_path (str, optional): Path to save the visualizations. Defaults to None.

    Returns:
        Tuple[np.ndarray, str]: Video visualization and the name of the output video.
    """
    out_vid = []
    # RGB VIS
    rgb_3thw = (batch["rgb_b3thw"] * batch["rgb_std_b3111"] + batch["rgb_mean_b3111"])[0]
    rgb_thw3 = rgb_3thw.cpu().numpy().transpose((1, 2, 3, 0))
    out_vid.append(rgb_thw3)
    seq_name = batch["seq_name"][0]

    if out_path is not None:
        os.makedirs(out_path, exist_ok=True)

    for task in tasks:
        if task == "depth":
            vis_min_depth, vis_max_depth = 0.05, 20.0
            depth_est_1thw = apply_fn(out["depth_est_b1thw"][0], fn_type="linear")
            depth_range = (
                max(torch.min(depth_est_1thw[depth_est_1thw > 0]).item(), vis_min_depth),
                min(torch.max(depth_est_1thw[depth_est_1thw > 0]).item(), vis_max_depth),
            )
            depth_est_1thw = torch.clamp(depth_est_1thw, min=depth_range[0], max=depth_range[1])
            depth_est_vis, _, _ = colormap_image(depth_est_1thw, vmin=depth_range[0], vmax=depth_range[1])
            depth_est_vis = depth_est_vis.cpu().numpy().transpose((1, 2, 3, 0))
            out_vid.append(depth_est_vis)

        if task == "flow_2d_backward":
            # FLOW VIS
            flow_2d_backward_est_b2thw = out["flow_2d_backward_est_b2thw"].cpu()
            bflow_est_vis_b3thw, flow_bounds = flow_video_to_color_with_bounds(
                flow_2d_backward_est_b2thw, None, max_flow_mag=25.0
            )
            bflow_est_vis_thw3 = bflow_est_vis_b3thw[0].numpy().transpose((1, 2, 3, 0))
            bflow_est_vis_thw3 = bflow_est_vis_thw3.astype(np.float32)
            out_vid.append(bflow_est_vis_thw3)

        if task == "dyn_mask":
            # DYN MASK EST
            dyn_mask_est_1thw = out["dyn_mask_est_b1thw"][0]
            dyn_mask_est_1thw = apply_fn(dyn_mask_est_1thw, fn_type="sigmoid")
            vis_thr = 0.85
            dyn_mask_est_1thw = (dyn_mask_est_1thw > vis_thr).to(dtype=torch.float32)
            dyn_mask_est_thw3 = dyn_mask_est_1thw[0, ..., None].repeat(1, 1, 1, 3).cpu().numpy()
            dyn_mask_est_thw3 = dyn_mask_est_thw3.astype(np.float32)
            out_vid.append(dyn_mask_est_thw3)

        if task == "track_2d":
            # TRACK VIS
            out_vis_2d_thw3 = visualize_2d_tracks(
                batch, out, vis_fn_est="sigmoid", tracks_leave_trace=16, vis_thr=0.5
            )
            out_vid.append(out_vis_2d_thw3)

    out_vid = np.concatenate(out_vid, axis=-2)

    if out_path is not None:
        out_vid_name = os.path.join(out_path, f"{seq_name}.mp4")
        media.write_video(out_vid_name, out_vid, fps=15)
    else:
        out_vid_name = None

    return out_vid, out_vid_name


def generate_4D_visualization(batch, out, tasks, out_path):
    """Process and visualize 4D reconstruction.

    Args:
        batch (Dict[str, Any]): Batch of data to process and visualize.
        out (Dict[str, Any]): Output of the model.
        tasks (List[str]): Tasks to visualize.
        out_path (str, optional): Path to save the visualizations.
    """
    B, _, T, H, W = batch["rgb_b3thw"].shape
    assert "depth" in tasks and "camray" in tasks, "Tasks must include depth, camray"
    assert B == 1, "Current implementation supports only batch size 1"
    seq_name = batch["seq_name"][0]
    device = batch["rgb_b3thw"].device
    dtype = batch["rgb_b3thw"].dtype

    out_path = os.path.join(out_path, seq_name)
    os.makedirs(out_path, exist_ok=True)

    if "traj3d_est_b16t" in out.keys():
        batch["intrinsics_b44t"] = out["traj3d_intrinsics_est_b16t"].reshape(1, 4, 4, T)

    if "camray_est_b6thw" in out.keys():
        intrinsics_norm_b44t = normalize_intrinsics(batch["intrinsics_b44t"], H, W).to(
            dtype=dtype, device=device
        )
        extrinsics_est_b44t, _ = rays_to_cameras(
            camray_b6thw=out["camray_est_b6thw"], intrinsics_b44t=intrinsics_norm_b44t, ctr_only=False
        )
    else:
        extrinsics_est_b44t = torch.linalg.inv(
            out["traj3d_est_b16t"].permute(0, 2, 1).reshape(1, T, 4, 4)
        ).permute(0, 2, 3, 1)

    extrinsics_est_b44t = get_cam_T_ref(extrinsics_est_b44t, ref_idx=0)

    rgb_b3thw = batch["rgb_b3thw"] * batch["rgb_std_b3111"] + batch["rgb_mean_b3111"]

    # track3d
    if "track_2d" in tasks:
        vis_thr = 0.75

        fix_scale = True
        track_2d_vis_est_bn1t = apply_fn(out["track_2d_vis_est_bn1t"], "sigmoid").clone()
        track_2d_depth_est_bn1t = apply_fn(out["track_2d_depth_est_bn1t"], "linear").clone()
        track_2d_traj_est_bn2t = out["track_2d_traj_est_bn2t"].clone()

        if fix_scale:
            traj_norm = out["track_2d_traj_est_bn2t"][0].clone()
            traj_norm[:, 0, :] = traj_norm[:, 0, :] / (W - 1) * 2 - 1
            traj_norm[:, 1, :] = traj_norm[:, 1, :] / (H - 1) * 2 - 1
            traj_sampled_depth_nt = torch.nn.functional.grid_sample(
                out["depth_est_b1thw"][0].permute(1, 0, 2, 3).to(dtype=torch.float32),
                traj_norm.permute(2, 0, 1).unsqueeze(2),
                mode="nearest",
                align_corners=False,
            )[:, 0, :, 0].permute(1, 0)

            vis_est_nt = track_2d_vis_est_bn1t[0, :, 0] > vis_thr
            traj_sampled_depth_good = traj_sampled_depth_nt[vis_est_nt > 0]
            track_2d_depth_good = track_2d_depth_est_bn1t[0, :, 0][vis_est_nt > 0]
            scale = torch.median(traj_sampled_depth_good / track_2d_depth_good)
            track_2d_depth_est_bn1t = scale * track_2d_depth_est_bn1t

        track_pc_list = generate_3d_track_point_clouds(
            track_2d_traj_est_bn2t,
            track_2d_depth_est_bn1t,
            track_2d_vis_est_bn1t,
            batch["intrinsics_b44t"].clone(),
            extrinsics_est_b44t.clone(),
            vis_thr=vis_thr,
            tracks_leave_trace=16,
            sort_points_by_height=True,
        )[0]

    camera_traj_mesh_list = generate_video_camera_trajectory(extrinsics_est_b44t.clone())[0]
    point_clouds_list = generate_video_point_clouds(
        rgb_b3thw.clone(),
        out["depth_est_b1thw"].clone(),
        batch["intrinsics_b44t"].clone(),
        extrinsics_est_b44t.clone(),
    )[0]

    ply_paths = []
    for index in range(T):
        if "track_2d" not in tasks:
            ply_path = os.path.join(out_path, f"{index}_world.ply")
            o3d.io.write_point_cloud(ply_path, point_clouds_list[index])
            ply_path_cam = ply_path.replace(".ply", "_cam_mesh.ply")
            o3d.io.write_triangle_mesh(ply_path_cam, camera_traj_mesh_list[index])
            ply_paths.append(
                {
                    "name": f"{seq_name}_{index}",
                    "pc_depth": ply_path,
                    "mesh_cam": ply_path_cam,
                }
            )
        else:
            ply_path = os.path.join(out_path, f"{index}_world.ply")
            ply_path_cam = ply_path.replace(".ply", "_cam_mesh.ply")
            o3d.io.write_triangle_mesh(ply_path_cam, camera_traj_mesh_list[index])
            ply_path_track_depth = ply_path.replace(".ply", "_track_depth_pc.ply")
            o3d.io.write_point_cloud(
                ply_path_track_depth,
                point_clouds_list[index] + track_pc_list[index],
            )
            ply_paths.append(
                {
                    "name": f"{seq_name}_{index}",
                    "pc_depth_track": ply_path_track_depth,
                    "mesh_cam": ply_path_cam,
                }
            )

    return ply_paths


##############################################################################################
# DEPTHMAP VISUALIZATION
##############################################################################################
def colormap_image(
    image_1thw,
    mask_1thw=None,
    invalid_color=(0.0, 0, 0.0),
    flip=True,
    vmin=None,
    vmax=None,
    return_vminvmax=False,
    colormap="turbo",
):
    """
    Colormaps a one channel tensor using a matplotlib colormap.

    Args:
        image_1thw: the tensor to colomap.
        mask_1thw: an optional float mask where 1.0 donates valid pixels.
        colormap: the colormap to use. Default is turbo.
        invalid_color: the color to use for invalid pixels.
        flip: should we flip the colormap? True by default.
        vmin: if provided uses this as the minimum when normalizing the tensor.
        vmax: if provided uses this as the maximum when normalizing the tensor.
            When either of vmin or vmax are None, they are computed from the
            tensor.
        return_vminvmax: when true, returns vmin and vmax.

    Returns:
        image_cm_3thw: image of the colormapped tensor.
        vmin, vmax: returned when return_vminvmax is true.


    """
    valid_vals = image_1thw if mask_1thw is None else image_1thw[mask_1thw.bool()]
    if vmin is None:
        vmin = valid_vals.min()
    if vmax is None:
        vmax = valid_vals.max()

    cmap = torch.Tensor(plt.cm.get_cmap(colormap)(torch.linspace(0, 1, 256))[:, :3]).to(  # type: ignore
        image_1thw.device
    )
    if flip:
        cmap = torch.flip(cmap, (0,))

    t, h, w = image_1thw.shape[1:]

    image_norm_1thw = (image_1thw - vmin) / ((vmax - vmin) * 1.05)
    image_int_1thw = (torch.clamp(image_norm_1thw * 255, 0, 255)).byte().long()

    image_cm_3thw = cmap[image_int_1thw.flatten(start_dim=1)].permute([0, 2, 1]).view([-1, t, h, w])

    if mask_1thw is not None:
        mask_1thw = mask_1thw.float()
        invalid_color = torch.Tensor(invalid_color).view(3, 1, 1, 1).to(image_1thw.device)
        image_cm_3thw = image_cm_3thw * mask_1thw + invalid_color * (1 - mask_1thw)

    return image_cm_3thw, vmin, vmax


##############################################################################################
# FLOW VISUALIZATION
##############################################################################################
def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_color_with_bounds(flow_uv, rad_max=None, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, "input flow must have three dimensions"
    assert flow_uv.shape[2] == 2, "input flow must have shape [H,W,2]"
    if clip_flow is not None:
        # flow_uv = np.clip(flow_uv, 0, clip_flow)
        flow_uv = np.clip(flow_uv, -clip_flow, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    if rad_max is None:
        rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def flow_video_to_color_with_bounds(flow_uv_b2thw, flow_bounds=None, max_flow_mag=-1.0):

    # assert flow_bounds is None
    B, _, T, H, W = flow_uv_b2thw.shape
    flow_viz = np.zeros((B, 3, T, H, W))

    flow_bounds_new = []
    for b in range(B):
        rad_max = (
            torch.max(torch.sqrt(torch.square(flow_uv_b2thw[0, 0]) + np.square(flow_uv_b2thw[0, 1])))
            .cpu()
            .item()
        )
        rad_max = min(max_flow_mag, rad_max) if max_flow_mag > 0 else rad_max
        rad_max = rad_max if flow_bounds is None else flow_bounds[b]
        for t in range(T):
            flow_i = flow_uv_b2thw[b, :, t].permute(1, 2, 0).detach().cpu().numpy()
            flow_viz_i = flow_to_color_with_bounds(flow_i, rad_max=rad_max, clip_flow=rad_max / np.sqrt(2))
            flow_viz[b, :, t] = flow_viz_i.transpose(2, 0, 1)
        flow_bounds_new.append(rad_max)

    flow_viz = torch.from_numpy(flow_viz).to(flow_uv_b2thw.device).to(flow_uv_b2thw.dtype) / 255.0

    return flow_viz, flow_bounds_new


##############################################################################################
# 2D TRACK VISUALIZATION
##############################################################################################
def visualize_2d_tracks(
    batch,
    out,
    vis_fn_est="sigmoid",
    tracks_leave_trace=16,
    vis_thr=0.75,
):
    """Function to visualize 2D tracks.

    Args:
        batch (Dict[str, Any]): Batch of data to process and visualize.
        out (Dict[str, Any]): Output of the model.
        vis_fn_est (str, optional): Function to apply to the visibility estimate. Defaults to "sigmoid".
        tracks_leave_trace (int, optional): Number of frames to leave the trace of the tracks. Defaults to 16.
        vis_thr (float, optional): Threshold for the visibility estimate. Defaults to 0.75.

    Returns:
        np.ndarray: Video visualization of the 2D tracks.
    """
    rgb_b3thw = batch["rgb_b3thw"] * batch["rgb_std_b3111"] + batch["rgb_mean_b3111"]
    sorted_indices = torch.argsort(batch["track_2d_traj_bn2t"][0, :, 1, 0])  # Sort points over height

    track_2d_vis_est_bn1t = apply_fn(out["track_2d_vis_est_bn1t"], vis_fn_est)
    track_2d_vis_est_bn1t = track_2d_vis_est_bn1t[:, sorted_indices, :, :]
    track_2d_traj_est_bn2t = out["track_2d_traj_est_bn2t"][:, sorted_indices, :, :]

    vis_bnt = track_2d_vis_est_bn1t[..., 0, :]
    vis_bnt = vis_bnt.cpu().numpy() > vis_thr
    vis_tn = einops.rearrange(vis_bnt[0], "n t -> t n")

    rgb_thw3 = einops.rearrange(rgb_b3thw[0].clone(), "c t h w -> t h w c")
    rgb_thw3 = torch.repeat_interleave(torch.mean(rgb_thw3, dim=-1, keepdims=True), 3, dim=-1).cpu().numpy()  # type: ignore
    track_2d_traj_tn2 = einops.rearrange(track_2d_traj_est_bn2t[0], "n c t -> t n c").cpu().numpy()
    out_vis_2d_thw3 = plot_2d_tracks(
        rgb_thw3, track_2d_traj_tn2, vis_tn, tracks_leave_trace=tracks_leave_trace
    )
    return out_vis_2d_thw3


def plot_2d_tracks(video, points, visibles, infront_cameras=None, tracks_leave_trace=16, show_occ=False):
    """Visualize 2D point trajectories."""
    num_frames, num_points = points.shape[:2]

    # Precompute colormap for points
    # color_map = matplotlib.colormaps.get_cmap('hsv') # AB
    color_map = pltcm.get_cmap("hsv")  # AB
    cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)
    point_colors = np.zeros((num_points, 3))
    for i in range(num_points):
        point_colors[i] = np.array(color_map(cmap_norm(i)))[:3]  # * 255

    if infront_cameras is None:
        infront_cameras = np.ones_like(visibles).astype(bool)

    frames = []
    for t in range(num_frames):
        frame = video[t].copy()

        # Draw tracks on the frame
        line_tracks = points[max(0, t - tracks_leave_trace) : t + 1]
        line_visibles = visibles[max(0, t - tracks_leave_trace) : t + 1]
        line_infront_cameras = infront_cameras[max(0, t - tracks_leave_trace) : t + 1]
        for s in range(line_tracks.shape[0] - 1):
            img = frame.copy()

            for i in range(num_points):
                if line_visibles[s, i] and line_visibles[s + 1, i]:  # visible
                    x1, y1 = int(round(line_tracks[s, i, 0])), int(round(line_tracks[s, i, 1]))
                    x2, y2 = int(round(line_tracks[s + 1, i, 0])), int(round(line_tracks[s + 1, i, 1]))
                    cv2.line(frame, (x1, y1), (x2, y2), point_colors[i], 1, cv2.LINE_AA)
                elif show_occ and line_infront_cameras[s, i] and line_infront_cameras[s + 1, i]:  # occluded
                    x1, y1 = int(round(line_tracks[s, i, 0])), int(round(line_tracks[s, i, 1]))
                    x2, y2 = int(round(line_tracks[s + 1, i, 0])), int(round(line_tracks[s + 1, i, 1]))
                    cv2.line(frame, (x1, y1), (x2, y2), point_colors[i], 1, cv2.LINE_AA)

            alpha = (s + 1) / (line_tracks.shape[0] - 1)
            frame = cv2.addWeighted(frame, alpha, img, 1 - alpha, 0)

        # Draw end points on the frame
        for i in range(num_points):
            if visibles[t, i]:  # visible
                x, y = int(round(points[t, i, 0])), int(round(points[t, i, 1]))
                cv2.circle(frame, (x, y), 2, point_colors[i], -1)
            elif show_occ and infront_cameras[t, i]:  # occluded
                x, y = int(round(points[t, i, 0])), int(round(points[t, i, 1]))
                cv2.circle(frame, (x, y), 2, point_colors[i], 1)

        frames.append(frame)
    frames = np.stack(frames)
    return frames


##############################################################################################
# CAMERA VISUALIZATION
##############################################################################################
def create_camera_frustum(
    fov_vertical=45,
    near=0.01,
    far=0.65,
    aspect_ratio=1,  # 16 / 9,
    camera_position=np.array([0, 0, 0]),
) -> o3d.geometry.TriangleMesh:
    """
    Create a camera frustum mesh using Open3D.

    Args:
        fov_horizontal: Horizontal field of view in degrees
        fov_vertical: Vertical field of view in degrees
        near: Near plane distance
        far: Far plane distance
        aspect_ratio: Width/height ratio
        camera_position: Position of the camera in 3D space

    Returns:
        open3d.geometry.TriangleMesh: The frustum mesh
    """

    # Convert degrees to radians
    fov_v_rad = np.radians(fov_vertical)

    # Calculate frustum dimensions
    near_height = 2 * near * np.tan(fov_v_rad / 2)
    near_width = near_height * aspect_ratio

    far_height = 2 * far * np.tan(fov_v_rad / 2)
    far_width = far_height * aspect_ratio

    # Define vertices for the frustum (8 vertices)
    vertices = np.array(
        [
            # Near plane (4 vertices)
            [-near_width / 2, -near_height / 2, near],  # Bottom-left
            [near_width / 2, -near_height / 2, near],  # Bottom-right
            [near_width / 2, near_height / 2, near],  # Top-right
            [-near_width / 2, near_height / 2, near],  # Top-left
            # Far plane (4 vertices)
            [-far_width / 2, -far_height / 2, far],  # Bottom-left
            [far_width / 2, -far_height / 2, far],  # Bottom-right
            [far_width / 2, far_height / 2, far],  # Top-right
            [-far_width / 2, far_height / 2, far],  # Top-left
        ]
    )

    # Translate vertices to camera position
    vertices += camera_position

    # Define triangles for the frustum (12 triangles total)
    # Each face has 2 triangles, and we have 6 faces
    triangles = np.array(
        [
            # Near face
            [0, 1, 2],
            [0, 2, 3],
            # Far face
            [4, 6, 5],
            [4, 7, 6],
            # Left face
            [0, 3, 7],
            [0, 7, 4],
            # Right face
            [1, 5, 6],
            [1, 6, 2],
            # Bottom face
            [0, 4, 5],
            [0, 5, 1],
            # Top face
            [3, 2, 6],
            [3, 6, 7],
        ]
    )

    # Create the mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Generate a random color for the mesh
    # random_color = np.random.rand(3)
    random_color = np.array([1.0, 0.5, 0.5])
    mesh.paint_uniform_color(random_color)

    # Compute normals for proper rendering
    mesh.compute_vertex_normals()

    return mesh


def generate_video_camera_trajectory(extrinsics_b44t: torch.Tensor) -> List[List[o3d.geometry.TriangleMesh]]:
    """Visualize camera trajectory"""
    extrinsics_b44t = extrinsics_b44t.cpu().numpy()
    B, _, _, T = extrinsics_b44t.shape
    out_o3dmesh_2dlist = []
    for b in range(B):
        out = []
        for index in range(T):
            extrinsics_44 = extrinsics_b44t[b, ..., index]
            world_T_cam = np.linalg.inv(extrinsics_44)
            # generte camera frustum
            frustum_mesh = create_camera_frustum()
            cam_vertices_n3 = np.asarray(frustum_mesh.vertices)
            cam_vertices_n4 = np.concatenate(
                (cam_vertices_n3, np.ones_like(cam_vertices_n3[..., :1])), axis=-1
            )
            cam_vertices_n4 = cam_vertices_n4 @ world_T_cam.T
            cam_vertices_n3 = cam_vertices_n4[..., :3]
            frustum_mesh.vertices = o3d.utility.Vector3dVector(cam_vertices_n3)
            out.append(frustum_mesh)
        out_o3dmesh_2dlist.append(out)
    return out_o3dmesh_2dlist


##############################################################################################
# 3D POINT CLOUD VISUALIZATION
##############################################################################################


def generate_video_point_clouds(
    rgb_b3thw: torch.Tensor,
    depth_b1thw: torch.Tensor,
    intrinsics_b44t: torch.Tensor,
    extrinsics_b44t: torch.Tensor,
) -> List[List[o3d.geometry.PointCloud]]:
    """Generate point cloud"""
    B, _, T, H, W = rgb_b3thw.shape

    world_T_cam_b44t = torch.linalg.inv(extrinsics_b44t.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    point_map_b3thw = generate_point_map(depth_b1thw, intrinsics_b44t, world_T_cam_b44t)
    rgb_btn3 = rgb_b3thw.reshape(B, 3, T, H * W).permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    point_map_btn3 = (
        point_map_b3thw.reshape(B, 3, T, H * W).permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    )

    clouds_list = []

    for b in range(B):
        clouds = []
        for index in range(T):
            points = point_map_btn3[b, index]
            colors = rgb_btn3[b, index]
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(colors)
            clouds.append(cloud)
        clouds_list.append(clouds)

    return clouds_list


##############################################################################################
# 3D TRACK POINT MAP VISUALIZATION
##############################################################################################


def generate_3d_track_point_clouds(
    track_2d_traj_bn2t: torch.Tensor,
    track_2d_depth_bn1t: torch.Tensor,
    track_2d_vis_bn1t: torch.Tensor,
    intrinsics_b44t: torch.Tensor,
    extrinsics_b44t: torch.Tensor,
    vis_thr: float = 0.5,
    tracks_leave_trace: int = 48,
    sort_points_by_height: bool = True,
) -> List[List[o3d.geometry.PointCloud]]:
    """
    Generates 3D track point clouds from 2D track, depth and camear information
    Note: The current implementation is CPU-based and may be slow for large number of tracks.

    Args:
        track_2d_traj_bn2t (torch.Tensor): 2D track trajectory
        track_2d_depth_bn1t (torch.Tensor): 2D track depth
        track_2d_vis_bn1t (torch.Tensor): 2D track visibility
        intrinsics_b44t (torch.Tensor): Camera intrinsics
        extrinsics_b44t (torch.Tensor): Camera to world transformation
        vis_thr (float, optional): _description_. Defaults to 0.5.
        tracks_leave_trace (int, optional): _description_. Defaults to 48.
        sort_points_by_height (bool, optional): Sort points by height for pretty visualization. Defaults to True.

    Returns:
        List[List[o3d.geometry.PointCloud]]: 3D track point clouds
    """

    B, N, _, T = track_2d_depth_bn1t.shape
    if sort_points_by_height:
        sorted_indices_bn = torch.argsort(track_2d_traj_bn2t[:, :, 1, 0], dim=1)
        gather_idx = sorted_indices_bn[:, :, None, None].expand(-1, -1, 2, T)
        track_2d_traj_bn2t = torch.gather(track_2d_traj_bn2t, 1, gather_idx)
        track_2d_depth_bn1t = torch.gather(track_2d_depth_bn1t, 1, gather_idx[:, :, :1, :])
        track_2d_vis_bn1t = torch.gather(track_2d_vis_bn1t, 1, gather_idx[:, :, :1, :])

    world_T_cam_b44t = torch.linalg.inv(extrinsics_b44t.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    XYZ_bn3t = generate_3d_track_point_map(
        track_2d_traj_bn2t, track_2d_depth_bn1t, intrinsics_b44t, world_T_cam_b44t
    )
    XYZ_btn3 = einops.rearrange(XYZ_bn3t, "b n k t -> b t n k").cpu().numpy()
    track_2d_vis_bn1t = track_2d_vis_bn1t.cpu().numpy()
    color_map = pltcm.get_cmap("hsv")
    cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=N - 1)

    point_clouds_list = []
    for b in range(B):
        point_clouds = []
        for t in range(T):
            track_points_3D = []
            track_points_color = []
            for i in range(N):
                if track_2d_vis_bn1t[b, i, 0, t] > vis_thr:
                    color = color_map(cmap_norm(i))
                    line_points = XYZ_btn3[b, max(0, t - tracks_leave_trace) : t + 1, i]
                    # interpolate line points
                    if line_points.shape[0] == 1:
                        track_points_3D.append(line_points)
                        track_points_color.append(
                            np.array(color)[None, :3].repeat(line_points.shape[0], axis=0)
                        )
                    else:
                        for k in range(line_points.shape[0] - 1):
                            start = line_points[k]
                            stop = line_points[k + 1]
                            alpha = np.linspace(0, 1, 20)
                            track_points_3D.append(start[None, :] + (stop - start)[None, :] * alpha[:, None])
                            track_points_color.append(
                                np.array(color)[None, :3].repeat(track_points_3D[-1].shape[0], axis=0)
                            )
            cloud = o3d.geometry.PointCloud()
            if len(track_points_3D) > 0:
                track_points_3D = np.concatenate(track_points_3D, axis=0)
                track_points_color = np.concatenate(track_points_color, axis=0)
                cloud.points = o3d.utility.Vector3dVector(track_points_3D)
                cloud.colors = o3d.utility.Vector3dVector(track_points_color)
            point_clouds.append(cloud)
        point_clouds_list.append(point_clouds)

    return point_clouds_list
