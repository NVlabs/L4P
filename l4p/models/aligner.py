# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from skimage.measure import ransac
from skimage.transform import SimilarityTransform

from l4p.utils.geometry_utils import generate_point_map
from l4p.utils.misc import safe_inverse


class WindowOverlapAligner(ABC):
    @abstractmethod
    def solve(self, pred, target, intrinsics, img_info):
        pass

    @abstractmethod
    def apply(self, pred):
        pass


class LstSqAffineAligner(WindowOverlapAligner):
    """
    Aligns two tensors using scale and shift estimation.
    Used for depth estimation.
    """

    def __init__(self, pre_post_fn: Optional[str] = "identity") -> None:
        if pre_post_fn == "identity":
            self.pre_fn = self.post_fn = lambda x: x
        elif pre_post_fn == "inverse":
            self.pre_fn = self.post_fn = safe_inverse
        elif pre_post_fn is None:
            self.pre_fn = self.post_fn = lambda x: x
        else:
            raise ValueError(f"Unknown pre_post_fn: {pre_post_fn}")

    def solve(self, pred, target, intrinsics, img_info, pred_conf=None, target_conf=None):
        pred = self.pre_fn(pred)
        target = self.pre_fn(target)

        bs = pred.shape[0]
        pred_rs = pred.reshape(bs, -1, 1)  # B x (...) x 1
        target_rs = target.reshape(bs, -1, 1)  # B x (...) x 1

        _ones = torch.ones_like(pred_rs)
        A = torch.cat([pred_rs, _ones], dim=-1)  # B x (...) x 2
        self.sol = torch.linalg.lstsq(A.float(), target_rs.float(), rcond=None).solution[..., 0]  # B x 2
        self.sol = self.sol.to(pred.dtype)

    def apply(self, pred):
        scale = self.sol[:, 0].reshape(self.sol.shape[0], *((1,) * (pred.ndim - 1)))
        shift = self.sol[:, 1].reshape(self.sol.shape[0], *((1,) * (pred.ndim - 1)))

        pred = self.pre_fn(pred)
        pred = scale * pred + shift
        pred = self.post_fn(pred)

        return pred


class LinearAligner(WindowOverlapAligner):
    """
    Aligns two tensors using scale estimation.
    Provides option to use mean or median scale.
    Used for depth estimation.
    """

    def __init__(self, pre_post_fn: Optional[str] = "identity", method: str = "mean") -> None:
        if pre_post_fn == "identity":
            self.pre_fn = self.post_fn = lambda x: x
        elif pre_post_fn == "inverse":
            self.pre_fn = self.post_fn = safe_inverse
        elif pre_post_fn is None:
            self.pre_fn = self.post_fn = lambda x: x
        else:
            raise ValueError(f"Unknown pre_post_fn: {pre_post_fn}")

        if method not in ["mean", "median"]:
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.sol_default = torch.Tensor([1.0])

    def solve(self, pred, target, intrinsics, img_info, pred_conf=None, target_conf=None):
        pred = self.pre_fn(pred)
        target = self.pre_fn(target)

        bs = pred.shape[0]
        pred_rs = pred.reshape(bs, -1)  # B x (...)
        target_rs = target.reshape(bs, -1)  # B x (...)

        # Compute ratios between target and prediction
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        ratios = target_rs / (pred_rs + eps)  # B x (...)

        if self.method == "mean":
            self.sol = torch.mean(ratios, dim=1)  # B
        else:  # median
            self.sol = torch.median(ratios, dim=1).values  # B

        self.sol = self.sol.to(pred.dtype)

    def apply(self, pred):
        scale = self.sol.reshape(self.sol.shape[0], *((1,) * (pred.ndim - 1)))

        pred = self.pre_fn(pred)
        pred = scale * pred
        pred = self.post_fn(pred)

        return pred


def get_similarity_3d_transform(src, dst, min_samples=5, reprojection_threshold=0.1, confidence=0.99):
    """Get similarity transform between two point clouds
    RANSAC is used to find the best similarity transform.
    dst = s * R @ src + t, where s is the scale, R is the rotation, and t is the translation.
    The similarity transform T = [s*R | t].

    Args:
        src (np.ndarray): Source point cloud
        dst (np.ndarray): Destination point cloud
        min_samples (int, optional): Minimum number of samples to use for RANSAC. Defaults to 5.
        reprojection_threshold (float, optional): Reprojection threshold for RANSAC. Defaults to 0.1.
        confidence (float, optional): Confidence level for RANSAC. Defaults to 0.99.

    Returns:
        Rt (dict): Similarity transform parameters
        inliers (np.ndarray): Inliers mask (boolean)
    """

    model_robust, inliers = ransac(
        (src, dst),
        model_class=SimilarityTransform,
        min_samples=min_samples,
        residual_threshold=reprojection_threshold,
        stop_probability=confidence,
        max_trials=100,
    )

    Rt = {
        "T": model_robust.params,
        "R": model_robust.rotation / model_robust.scale,
        "t": model_robust.translation,
        "s": model_robust.scale,
    }

    return Rt, inliers


class KabaschUmeyama3DAligner(WindowOverlapAligner):
    """
    Jointly aligns the point maps (estimated from depthmaps and cameray trajectories) from overlapping windows.
    Uses similarity transform to align the point maps.
    Note: Current implementation is CPU based and hence might be slow.
    """

    def __init__(
        self,
        calc_scale: bool = True,
    ) -> None:
        self.rel_T_b44 = None
        self.calc_scale = calc_scale
        self.min_samples = 10
        self.reprojection_threshold = 0.01
        self.confidence = 0.99
        self.frame_sample_step = 3
        self.point_sample_ratio = 0.1

    def solve(self, pred, target, img_info):
        task_names = pred.keys()
        assert "camray" in task_names, "camray is required for KabaschUmeyama3DAligner"
        assert "depth" in task_names, "depth is required for KabaschUmeyama3DAligner"

        device = pred["depth"].device
        dtype = pred["depth"].dtype
        bs, _, T, H, W = pred["depth"].shape

        # compute the depth range and scale the reprojection threshold accordingly
        depth_range = torch.quantile(pred["depth"].reshape(bs, -1).to(torch.float32), 0.98, dim=-1)
        reprojection_threshold = (depth_range * self.reprojection_threshold).cpu().numpy()

        # compute point maps from depth and cameray trajectories
        point_map_pred_b3thw = generate_point_map(
            depth_b1thw=pred["depth"][:, :, :: self.frame_sample_step, :, :],
            intrinsics_b44t=pred["camray_intrinsics"].reshape(bs, 4, 4, -1)[
                :, :, :, :: self.frame_sample_step
            ],
            world_T_cam_b44t=pred["camray"].reshape(bs, 4, 4, -1)[:, :, :, :: self.frame_sample_step],
        )
        point_map_tgt_b3thw = generate_point_map(
            depth_b1thw=target["depth"][:, :, :: self.frame_sample_step, :, :],
            intrinsics_b44t=target["camray_intrinsics"].reshape(bs, 4, 4, -1)[
                :, :, :, :: self.frame_sample_step
            ],
            world_T_cam_b44t=target["camray"].reshape(bs, 4, 4, -1)[:, :, :, :: self.frame_sample_step],
        )

        rel_T_b44 = []  # to be applied to est traj first
        for sample_idx in range(bs):
            xyz_pred_n3 = (
                point_map_pred_b3thw[sample_idx].reshape(3, -1).detach().cpu().numpy().T.astype(np.float32)
            )
            xyz_tgt_n3 = (
                point_map_tgt_b3thw[sample_idx].reshape(3, -1).detach().cpu().numpy().T.astype(np.float32)
            )

            # sample random indices from n and use that to sample both pred and tgt
            if self.point_sample_ratio < 1:
                n = xyz_pred_n3.shape[0]
                random_idx = np.random.permutation(n)[: int(self.point_sample_ratio * n)]
                xyz_pred_n3 = xyz_pred_n3[random_idx]
                xyz_tgt_n3 = xyz_tgt_n3[random_idx]

            rel_T, inliers = get_similarity_3d_transform(
                xyz_pred_n3,
                xyz_tgt_n3,
                min_samples=self.min_samples,
                reprojection_threshold=reprojection_threshold[sample_idx],
                confidence=self.confidence,
            )
            rel_T_b44.append(rel_T)

        rel_T_b44 = {
            key: torch.from_numpy(np.stack([rel_T_b44[i][key] for i in range(bs)], axis=0)).to(
                device=device, dtype=dtype
            )
            for key in rel_T_b44[0].keys()
        }
        self.rel_T_b44 = rel_T_b44

    def apply(self, pred):
        # apply transformation
        task_names = pred.keys()

        bs, _, T, H, W = pred["depth"].shape

        assert self.rel_T_b44 is not None, "rel_T_b44 is not set"
        assert bs == len(self.rel_T_b44["T"])

        pred_new = {}
        for task_name in task_names:
            if task_name == "camray":
                pose_est_b44t = pred[task_name].reshape(bs, 4, 4, T)
                # Apply the similarity transform to the camera pose
                pose_est_b44t = torch.einsum("bij,bjkt->bikt", self.rel_T_b44["T"], pose_est_b44t)
                # Remove the scale factor to get rotation only
                # This scale should come from depth and so applied to depth
                pose_est_b44t[:, :3, :3] = pose_est_b44t[:, :3, :3] / self.rel_T_b44["s"]
                pred_new[task_name] = pose_est_b44t.reshape(bs, -1, T)
            elif task_name == "depth":
                pred_new[task_name] = pred[task_name] * self.rel_T_b44["s"]
            elif task_name == "camray_intrinsics_est":
                pred_new[task_name] = pred[task_name]
            else:
                raise ValueError(f"Unknown task name: {task_name}")

        return pred_new
