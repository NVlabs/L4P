# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

from typing import List, Tuple, Dict, Optional, Type, Literal
import torch
from l4p.utils.misc import apply_fn
from l4p.models.aligner import (
    WindowOverlapAligner,
    LstSqAffineAligner,
    LinearAligner,
    KabaschUmeyama3DAligner,
)
from l4p.utils.geometry_utils import rays_to_cameras, rays_to_cameras_and_intrinsics, normalize_intrinsics
from l4p.models.task_heads.dpt.dust3r.dpt_head import PixelwiseTaskWithDPT


class VideoMAEFlowDPTHead(torch.nn.Module):
    """2D Optical Flow DPT Head"""

    def __init__(
        self,
        task_name: str,
        out_nchan: int = 2,
        depth: int = 40,
        embed_dim: int = 1408,
        hooks_idx: Optional[List[int]] = None,
        actpost_scale_factors: Tuple[Tuple, ...] = ((1, 2, 2), (1, 1, 1), (0, 0, 0), (-1, -1, -1)),
        fusion_scale_factors: Tuple[Tuple, ...] = ((1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2)),
        output_size: Optional[Tuple[int, int, int]] = None,
        overlap_aligner_type: Optional[Type[WindowOverlapAligner]] = None,
        aligner_kwargs: Dict = {},
    ) -> None:
        super().__init__()

        l2 = depth
        feature_dim = 256
        last_dim = feature_dim // 2
        self.out_nchan = out_nchan
        if hooks_idx is None:
            hooks_idx = [l2 * 2 // 5, l2 * 3 // 5, l2 * 4 // 5, l2]
        layer_dims = [256, 512, 1024, 1024]
        is_use_conv3d = True
        self.task_name = task_name
        self.overlap_aligner_type = overlap_aligner_type
        self.aligner_kwargs = aligner_kwargs
        self.output_size = output_size
        self.task_suffix = f"b{out_nchan}thw"

        self.task_head = PixelwiseTaskWithDPT(
            num_channels=out_nchan,
            feature_dim=feature_dim,
            last_dim=last_dim,
            hooks_idx=hooks_idx,
            layer_dims=layer_dims,
            dim_tokens=[embed_dim, embed_dim, embed_dim, embed_dim],
            is_use_conv3d=is_use_conv3d,
            head_type="regression",
            actpost_scale_factors=actpost_scale_factors,
            fusion_scale_factors=fusion_scale_factors,
            output_size=self.output_size,
        )

    def forward(
        self,
        enc_features_bpc_list: List[torch.Tensor],
        img_info: Tuple[int, int, int] = (16, 224, 224),
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        task_out = self.task_head(enc_features_bpc_list, img_info)
        out = {f"{self.task_name}_est_{self.task_suffix}": task_out[:, : self.out_nchan]}
        return out

    def forward_windowed(
        self,
        enc_features_bpc_2dlist: List[List[torch.Tensor]],
        img_info: Tuple[int, int, int] = (16, 224, 224),
        time_strides: Optional[torch.Tensor] = None,
        intrinsics_b44t: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        if time_strides is None:
            return self.forward(
                enc_features_bpc_2dlist[0],
                img_info=img_info,
                intrinsics_b44t=intrinsics_b44t,
                **kwargs,
            )

        dtype = enc_features_bpc_2dlist[0][0].dtype
        device = enc_features_bpc_2dlist[0][0].device

        if self.output_size is None:
            window_size, H, W = img_info
        else:
            window_size, H, W = self.output_size
        T = int(time_strides[-1] + window_size)

        est_bktS = None

        for win_id in range(time_strides.shape[0]):
            kwargs["win_id"] = win_id
            curr_start_time_id = time_strides[win_id]
            # compute estimation for the current window
            curr_out = self.forward(
                enc_features_bpc_2dlist[win_id],
                img_info=img_info,
                intrinsics_b44t=intrinsics_b44t[..., curr_start_time_id : curr_start_time_id + window_size],
                **kwargs,
            )
            out = curr_out[f"{self.task_name}_est_{self.task_suffix}"]

            if est_bktS is None:
                # initialize the output buffer
                sz = list(out.shape)
                sz[2] = T
                est_bktS = torch.zeros(*sz, dtype=dtype, device=device)

            # perform alignment of the current window with the previous window
            if win_id > 0 and self.overlap_aligner_type is not None:
                aligner = self.overlap_aligner_type(**self.aligner_kwargs)
                overlap_sz = time_strides[win_id - 1] + window_size - curr_start_time_id

                aligner.solve(
                    out[:, :, :overlap_sz],
                    est_bktS[:, :, curr_start_time_id : curr_start_time_id + overlap_sz],
                    intrinsics_b44t[..., curr_start_time_id : curr_start_time_id + overlap_sz],  # type: ignore
                    img_info,
                )
                out = aligner.apply(out)

            # update the output buffer
            if self.task_name == "flow_2d_backward" and win_id > 0:
                # for flow_backward, first frame of window is not valid
                est_bktS[:, :, curr_start_time_id + 1 : curr_start_time_id + window_size] = out[:, :, 1:]  # type: ignore
            else:
                est_bktS[:, :, curr_start_time_id : curr_start_time_id + window_size] = out  # type: ignore

        final_out = {f"{self.task_name}_est_{self.task_suffix}": est_bktS}  # type: ignore
        return final_out


class VideoMAEDepthDPTHead(VideoMAEFlowDPTHead):
    """Depth DPT Head"""

    def __init__(
        self,
        task_name: str,
        out_nchan: int = 1,
        depth: int = 40,
        embed_dim: int = 1408,
        depth_fn: str = "linear",
        hooks_idx: Optional[List[int]] = None,
        align_window_overlap_fn: Optional[str] = None,
        align_type: Literal["linear", "affine"] = "affine",
    ) -> None:
        super().__init__(
            task_name,
            out_nchan,
            depth,
            embed_dim,
            hooks_idx,
            overlap_aligner_type=LstSqAffineAligner if align_type == "affine" else LinearAligner,
            aligner_kwargs=dict(pre_post_fn=align_window_overlap_fn),
        )
        self.depth_fn = depth_fn
        return

    def forward(
        self,
        enc_features_bpc_list: List[torch.Tensor],
        img_info: Tuple[int, int, int] = (16, 224, 224),
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        task_out = self.task_head(enc_features_bpc_list, img_info)
        depth = apply_fn(task_out[:, : self.out_nchan], fn_type=self.depth_fn)
        out = {f"{self.task_name}_est_{self.task_suffix}": depth}

        return out


class VideoMAEDynMaskDPTHead(VideoMAEFlowDPTHead):
    """Dynamic Mask DPT Head"""

    def __init__(
        self,
        task_name: str,
        out_nchan: int = 1,
        depth: int = 40,
        embed_dim: int = 1408,
        apply_fn: str = "linear",
        hooks_idx: Optional[List[int]] = None,
    ) -> None:
        super().__init__(
            task_name,
            out_nchan,
            depth,
            embed_dim,
            hooks_idx,
            overlap_aligner_type=None,
        )
        self.apply_fn = apply_fn
        return

    def forward(
        self,
        enc_features_bpc_list: List[torch.Tensor],
        img_info: Tuple[int, int, int] = (16, 224, 224),
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        dyn_mask = self.task_head(enc_features_bpc_list, img_info)
        dyn_mask[:, 0] = apply_fn(dyn_mask[:, 0], fn_type=self.apply_fn)
        out = {f"{self.task_name}_est_{self.task_suffix}": dyn_mask}
        return out


class VideoMAECameraDPTHead(VideoMAEFlowDPTHead):
    """Camera Ray DPT Head"""

    def __init__(
        self,
        task_name: str,
        out_nchan: int = 6,
        depth: int = 40,
        embed_dim: int = 1408,
        hooks_idx: Optional[List[int]] = None,
        actpost_scale_factors: Tuple[Tuple, ...] = ((1, 0, 0), (1, 0, 0), (0, 0, 0), (-1, -1, -1)),
        fusion_scale_factors: Tuple[Tuple, ...] = ((1, 1, 1), (1, 1, 1), (2, 1, 1), (2, 2, 2)),
        output_size: Optional[Tuple[int, int, int]] = (16, 16, 16),
    ) -> None:
        super().__init__(
            task_name,
            out_nchan,
            depth,
            embed_dim,
            hooks_idx,
            actpost_scale_factors,
            fusion_scale_factors,
            output_size,
        )
        return

    def forward(
        self,
        enc_features_bpc_list: List[torch.Tensor],
        img_info: Tuple[int, int, int] = (16, 224, 224),
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        rays = self.task_head(enc_features_bpc_list, img_info)
        out = {f"{self.task_name}_est_{self.task_suffix}": rays}
        return out


class VideoMAETraj3DDPTHead(VideoMAEFlowDPTHead):
    """
    3D Trajectory DPT Head
    Computes 6D Plucker ray map and extracts camera poses and intrinsics
    """

    def __init__(
        self,
        task_name: str,
        depth: int = 40,
        embed_dim: int = 1408,
        hooks_idx: Optional[List[int]] = None,
        actpost_scale_factors: Tuple[Tuple, ...] = ((1, 0, 0), (1, 0, 0), (0, 0, 0), (-1, -1, -1)),
        fusion_scale_factors: Tuple[Tuple, ...] = ((1, 1, 1), (1, 1, 1), (2, 1, 1), (2, 2, 2)),
        output_size: Optional[Tuple[int, int, int]] = (16, 16, 16),
        use_intrinsics: bool = True,
        fixed_intrinsics: bool = False,
    ) -> None:
        super().__init__(
            task_name,
            6,  # plucker coordinates
            depth,
            embed_dim,
            hooks_idx,
            actpost_scale_factors,
            fusion_scale_factors,
            output_size,
        )
        self.task_suffix = "b16t"
        self.use_intrinsics = use_intrinsics
        self.fixed_intrinsics = fixed_intrinsics
        self.first_window_intrinsics_b44t = None

        return

    def forward(
        self,
        enc_features_bpc_list: List[torch.Tensor],
        img_info: Tuple[int, int, int] = (16, 224, 224),
        intrinsics_b44t: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        T, H, W = img_info
        rays_est_b6thw = self.task_head(enc_features_bpc_list, img_info).to(dtype=torch.float32)
        # get pose (wTc) from rays
        intrinsics_est_b44t = None
        if not self.use_intrinsics and self.fixed_intrinsics:
            assert "win_id" in kwargs, "win_id is required when setting fixed intrinsics as True"
            win_id = kwargs["win_id"]
            if win_id == 0:
                self.first_window_intrinsics_b44t = None

        if self.use_intrinsics:
            # use the input intrinsics
            ext_est_b44t, _ = rays_to_cameras(
                rays_est_b6thw,
                normalize_intrinsics(intrinsics_b44t, H, W).to(dtype=torch.float32),
                ctr_only=False,
            )
        elif self.fixed_intrinsics:
            if self.first_window_intrinsics_b44t is None:
                # estimate the intrinsics for the first window
                ext_est_b44t, _, intrinsics_est_b44t = rays_to_cameras_and_intrinsics(
                    rays_est_b6thw,
                    ctr_only=False,
                    reproj_threshold=0.2,
                    output_size=(H, W),
                    fixed_intrinsics=True,
                )
                self.first_window_intrinsics_b44t = intrinsics_est_b44t.clone()
            else:
                # use the already estimated intrinsics
                ext_est_b44t, _ = rays_to_cameras(
                    rays_est_b6thw,
                    normalize_intrinsics(intrinsics_b44t, H, W).to(dtype=torch.float32),
                    ctr_only=False,
                )
                intrinsics_est_b44t = self.first_window_intrinsics_b44t.clone()

        else:
            # estimate per-frame intrinsics and extrinsics
            ext_est_b44t, _, intrinsics_est_b44t = rays_to_cameras_and_intrinsics(
                rays_est_b6thw,
                ctr_only=False,
                reproj_threshold=0.2,
                output_size=(H, W),
                fixed_intrinsics=False,
            )

        pose_est_b44t = torch.linalg.inv(ext_est_b44t.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        pose_est_b16t = pose_est_b44t.reshape(pose_est_b44t.shape[0], 16, T)
        out = {f"{self.task_name}_est_{self.task_suffix}": pose_est_b16t}
        if intrinsics_est_b44t is not None:
            intrinsics_est_b16t = intrinsics_est_b44t.reshape(intrinsics_est_b44t.shape[0], 16, T)
            out[f"{self.task_name}_intrinsics_est_{self.task_suffix}"] = intrinsics_est_b16t
        return out


########################################################################################
# JOINT DEPTH AND CAMERA ALIGNMENT
########################################################################################


def joint_windowed_estimation(
    task_names: List[str],
    task_heads: torch.nn.ModuleDict,
    enc_features_bpc_2dlist: List[List[torch.Tensor]],
    time_strides: Optional[torch.Tensor] = None,
    intrinsics_b44t: Optional[torch.Tensor] = None,
    img_info: Tuple[int, int, int] = (16, 224, 224),
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Joint windowed estimation for depth and camray.
    Converts depth and camra to point maps for overlapping regions and aligns using similarity transform.
    Uses KabaschUmeyama3DAligner for alignment.
    """

    out_all_tasks = {}

    if time_strides is None:
        for task_name in task_names:
            out_curr_task = task_heads[task_name].forward(
                enc_features_bpc_2dlist[0],
                img_info=img_info,
                intrinsics_b44t=intrinsics_b44t,
                **kwargs,
            )
            out_all_tasks.update(out_curr_task)
        return out_all_tasks

    dtype = enc_features_bpc_2dlist[0][0].dtype
    device = enc_features_bpc_2dlist[0][0].device

    window_size, _, _ = img_info
    T = int(time_strides[-1] + window_size)

    # define output buffers
    est_bktS = {task_name: None for task_name in task_names}
    est_bktS.update({f"{task_name}_intrinsics_est": None for task_name in ["camray"]})

    # loop over all the windows given by time_strides
    for win_id in range(time_strides.shape[0]):
        curr_start_time_id = time_strides[win_id]
        kwargs["win_id"] = win_id

        # perform per-task estimation for the current window
        curr_out_all_tasks = {}
        for task_name in task_names:
            curr_out = task_heads[task_name].forward(
                enc_features_bpc_2dlist[win_id],
                img_info=img_info,
                intrinsics_b44t=intrinsics_b44t[..., curr_start_time_id : curr_start_time_id + window_size],
                **kwargs,
            )
            # simply overwrite previous window results
            out = curr_out[f"{task_heads[task_name].task_name}_est_{task_heads[task_name].task_suffix}"]

            if task_name == "camray":
                if (
                    f"{task_heads[task_name].task_name}_intrinsics_est_{task_heads[task_name].task_suffix}"
                    in curr_out.keys()
                ):
                    out_intrinsics_est = curr_out[
                        f"{task_heads[task_name].task_name}_intrinsics_est_{task_heads[task_name].task_suffix}"
                    ]
                else:
                    out_intrinsics_est = torch.clone(
                        intrinsics_b44t[..., curr_start_time_id : curr_start_time_id + window_size]
                    ).reshape(1, 16, window_size)

            if est_bktS[task_name] is None:
                sz = list(out.shape)
                sz[2] = T
                est_bktS[task_name] = torch.zeros(*sz, dtype=dtype, device=device)
                if task_name == "camray":
                    if out_intrinsics_est is not None:
                        sz = list(out_intrinsics_est.shape)
                        sz[2] = T
                        est_bktS[f"{task_name}_intrinsics_est"] = torch.zeros(*sz, dtype=dtype, device=device)

            curr_out_all_tasks[task_name] = out
            if task_name == "camray":
                if out_intrinsics_est is not None:
                    curr_out_all_tasks[f"{task_name}_intrinsics_est"] = out_intrinsics_est

        # Perform alignment of all tasks
        if win_id > 0:
            aligner = KabaschUmeyama3DAligner()
            overlap_sz = time_strides[win_id - 1] + window_size - curr_start_time_id

            # get dicts for overlapping regions
            pred = {}
            target = {}
            for task_name in task_names:
                pred[task_name] = curr_out_all_tasks[task_name][:, :, :overlap_sz]
                target[task_name] = est_bktS[task_name][
                    :, :, curr_start_time_id : curr_start_time_id + overlap_sz
                ]

            curr_intrinsics = curr_out_all_tasks["camray_intrinsics_est"][:, :, :overlap_sz].reshape(
                1, 4, 4, overlap_sz
            )
            target["camray_intrinsics"] = est_bktS["camray_intrinsics_est"][
                :, :, curr_start_time_id : curr_start_time_id + overlap_sz
            ].reshape(1, 4, 4, overlap_sz)
            pred["camray_intrinsics"] = curr_intrinsics.clone()

            # joint alignment
            aligner.solve(pred, target, img_info)
            curr_out_all_tasks = aligner.apply(curr_out_all_tasks)

        for task_name in task_names:
            est_bktS[task_name][:, :, curr_start_time_id : curr_start_time_id + window_size] = curr_out_all_tasks[task_name]  # type: ignore
            if task_name == "camray":
                if f"{task_name}_intrinsics_est" in curr_out_all_tasks.keys():
                    est_bktS[f"{task_name}_intrinsics_est"][
                        :, :, curr_start_time_id : curr_start_time_id + window_size
                    ] = curr_out_all_tasks[
                        f"{task_name}_intrinsics_est"
                    ]  # type: ignore

    for task_name in task_names:
        key = f"{task_heads[task_name].task_name}_est_{task_heads[task_name].task_suffix}"
        out_all_tasks[key] = est_bktS[task_name]
        if task_name == "camray":
            if (
                f"{task_name}_intrinsics_est" in est_bktS.keys()
                and est_bktS[f"{task_name}_intrinsics_est"] is not None
            ):
                key_intrinsics_est = (
                    f"{task_heads[task_name].task_name}_intrinsics_est_{task_heads[task_name].task_suffix}"
                )
                out_all_tasks[key_intrinsics_est] = est_bktS[f"{task_name}_intrinsics_est"]

    return out_all_tasks
