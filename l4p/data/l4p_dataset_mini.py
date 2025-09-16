# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

from math import ceil
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Tuple, Dict, Optional, Any, List, Literal
from dataclasses import dataclass, asdict
from kornia.morphology import erosion


@dataclass(eq=False)
class L4PData:
    """
    Dataclass for storing video depth, flow, 3D tracks, etc.
    Note that the attribute names also show sizes.
    However, the batch dimension will only come after dataloader.
    """

    rgb_b3thw: torch.Tensor  # rgb data normalized to [0,1], fp32
    intrinsics_b44t: Optional[torch.Tensor] = None  # per image intrinsics, fp32
    extrinsics_b44t: Optional[torch.Tensor] = None  # extrinsics, fp32 # TODO: Add support in L4PDataset
    rel_pose_b6t: Optional[torch.Tensor] = None  # w_T_c relative to first frame, specified as xyz + rotvec
    flow_2d_backward_b2thw: Optional[torch.Tensor] = None  # flow backward in pixel space, fp32
    flow_2d_backward_valid_b2thw: Optional[torch.Tensor] = None  # valid mask, fp32 with vals {0,1}
    flow_2d_forward_b2thw: Optional[torch.Tensor] = None  # flow forward in pixel space, fp32 with vals {0,1}
    flow_2d_forward_valid_b2thw: Optional[torch.Tensor] = None  # valid mask, fp32 with vals {0,1}
    depth_b1thw: Optional[torch.Tensor] = None
    depth_valid_b1thw: Optional[torch.Tensor] = None  # valid mask, fp32 with vals {0,1}
    instanceseg_b1thw: Optional[torch.Tensor] = None  # instance seg
    dyn_mask_b1thw: Optional[torch.Tensor] = None  # dynamic motion mask
    dyn_mask_valid_b1thw: Optional[torch.Tensor] = None  # dynamic motion mask
    track_2d_traj_bn2t: Optional[torch.Tensor] = None  # track xy pixels for video of length t, fp32
    track_2d_depth_bn1t: Optional[torch.Tensor] = None  # track depth for video of length t, fp32
    track_2d_vis_bn1t: Optional[torch.Tensor] = None  # visibility information for track, bool tensor
    track_2d_valid_bn1t: Optional[torch.Tensor] = None  # valid information for track, bool tensor
    track_2d_pointquerries_bn3: Optional[torch.Tensor] = None  # pass point querries from dataset
    track_2d_pointlabels_bn: Optional[torch.Tensor] = None  # just pass ones for now
    dataset_name: Optional["str"] = None  # dataset name to be passed for dataset specific logging/vis
    seq_name: Optional["str"] = None  # seq name to be passed for dumping results with seq name


# Literals
ESTIMATION_DIRECTIONS = Literal[1, -1]


class L4PDataset(Dataset):
    default_sample_size = (16, 224, 224)

    def __init__(
        self,
        crop_size: Optional[Tuple[int, int, int]] = default_sample_size,
        track_2d_traj_per_sample: int = 128,
        track_2d_vis_thr: int = 4,
        track_2d_repeat_traj: bool = True,
        center_crop: bool = False,
        start_crop_time: bool = False,
        resize_size: Optional[Tuple[int, int] | int] = None,
        resize_mode: Dict[str, str] = {"rgb_b3thw": "trilinear"},
        estimation_directions: List[ESTIMATION_DIRECTIONS] = [1, -1],
        traj_sampling_window: Optional[List[int]] = None,
        length_mutiply_of: int = 8,
        track_2d_querry_sampling_version: None | Literal["uniform", "uniform_over_seg"] = None,
        track_2d_querry_sampling_spacing: float = 0.02,
        remove_queries_outside_bounds: bool = True,
        scaling_mode: Optional[Literal["avg_pointmapdist", "max_depth"]] = None,
    ) -> None:
        """L4P Based dataset

        Args:
            crop_size (Tuple[int, int, int], optional): crop to this size. Defaults to (16, 224, 224).
            track_2d_traj_per_sample (int, optional): number of tracks per video. Defaults to 128.
            track_2d_vis_thr (int, optional): criteria for selecting tracks. Defaults to 4.
            center_crop (bool, optional): centered cropping. Defaults to False.
            start_crop_time (bool, optional): t0 for temporal cropping is 0. Defaults to False.
            resize_factor (Tuple[float, float] | float, optional): resize factors for H and W. Defaults to (1.0, 1.0).
            resize_mode (Dict[str, str], optional): define a dict for overwritting default resize modes. Defaults to {"rgb_b3thw": "trilinear"}.
        """
        super(L4PDataset, self).__init__()

        self.crop_size = crop_size
        self.track_2d_traj_per_sample = track_2d_traj_per_sample
        self.track_2d_vis_thr = track_2d_vis_thr
        self.track_2d_repeat_traj = track_2d_repeat_traj
        self.center_crop = center_crop
        self.start_crop_time = start_crop_time
        if resize_size is not None:
            resize_size = (resize_size, resize_size) if not isinstance(resize_size, tuple) else resize_size
        self.resize_size = resize_size
        self.resize_mode = self.setup_resize_mode(resize_mode)
        self.estimation_directions = estimation_directions
        self.traj_sampling_window = traj_sampling_window
        self.length_multiply_of = length_mutiply_of
        self.track_2d_querry_sampling_version = track_2d_querry_sampling_version
        self.track_2d_querry_sampling_spacing = track_2d_querry_sampling_spacing
        self.remove_queries_outside_bounds = remove_queries_outside_bounds
        self.scaling_mode = scaling_mode

        self.input_mean = torch.Tensor([0.485, 0.456, 0.406]).to(dtype=torch.float32)
        self.input_std = torch.Tensor([0.229, 0.224, 0.225]).to(dtype=torch.float32)

    def setup_resize_mode(self, resize_mode: Dict[str, str]) -> Dict[str, str]:
        out = {
            "rgb_b3thw": "trilinear",
            "depth_b1thw": "nearest",  # most times its sparse so nearest
            "instanceseg_b1thw": "nearest",
            "flow_2d_backward_b2thw": "nearest",  # safe to us nearest as compared to trilinear
            "flow_2d_forward_b2thw": "nearest",  # safe to us nearest as compared to trilinear
            "flow_2d_backward_valid_b2thw": "nearest",
            "flow_2d_forward_valid_b2thw": "nearest",
            "depth_valid_b1thw": "nearest",
            "dyn_mask_b1thw": "nearest",
            "dyn_mask_valid_b1thw": "nearest",
        }
        for key in resize_mode:
            out[key] = resize_mode[key]
        return out

    def getitem_helper(self, index: int) -> L4PData:
        raise NotImplementedError

    def mirror_and_pad(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Mirrors the video in temporal dimension with special care for some keys"""
        out = {}
        for key in sample.keys():
            if not torch.is_tensor(sample[key]):
                continue
            if key == "flow_2d_backward_b2thw":
                out["flow_2d_backward_b2thw"] = torch.cat(
                    [
                        sample["flow_2d_backward_b2thw"],
                        torch.flip(sample["flow_2d_forward_b2thw"], dims=[1])[:, 1:],
                    ],
                    dim=1,
                )
                out["flow_2d_backward_valid_b2thw"] = torch.cat(
                    [
                        sample["flow_2d_backward_valid_b2thw"],
                        torch.flip(sample["flow_2d_forward_valid_b2thw"], dims=[1])[:, 1:],
                    ],
                    dim=1,
                )

            elif key == "flow_2d_forward_b2thw":
                out["flow_2d_forward_b2thw"] = torch.cat(
                    [
                        sample["flow_2d_forward_b2thw"],
                        torch.flip(sample["flow_2d_backward_b2thw"], dims=[1])[:, 1:],
                    ],
                    dim=1,
                )
                out["flow_2d_forward_valid_b2thw"] = torch.cat(
                    [
                        sample["flow_2d_forward_valid_b2thw"],
                        torch.flip(sample["flow_2d_backward_valid_b2thw"], dims=[1])[:, 1:],
                    ],
                    dim=1,
                )
            elif key in ["flow_2d_forward_valid_b2thw", "flow_2d_backward_valid_b2thw"]:
                # already handled previously
                continue
            elif key in [
                "rgb_b3thw",
                "depth_b1thw",
                "depth_valid_b1thw",
                "instanceseg_b1thw",
                "dyn_mask_b1thw",
                "dyn_mask_valid_b1thw",
            ]:
                out[key] = torch.cat([sample[key], torch.flip(sample[key], dims=[1])[:, 1:]], dim=1)
            elif key in [
                "track_2d_traj_bn2t",
                "track_2d_depth_bn1t",
                "track_2d_vis_bn1t",
                "track_2d_valid_bn1t",
                "intrinsics_b44t",
                "extrinsics_b44t",
                "rel_pose_b6t",
            ]:
                out[key] = torch.cat([sample[key], torch.flip(sample[key], dims=[-1])[..., 1:]], dim=-1)
            elif key in ["track_2d_pointquerries_bn3", "track_2d_pointlabels_bn"]:
                out[key] = sample[key]
            else:
                raise NotImplementedError

        return out

    def repeat_single_frame(self, sample: Dict[str, torch.Tensor], length: int) -> Dict[str, torch.Tensor]:
        """Extend single frame to video with special care for some keys"""
        out = {}
        for key in sample.keys():
            if not torch.is_tensor(sample[key]):
                continue
            if key in {
                "flow_2d_backward_b2thw",
                "flow_2d_forward_b2thw",
                "flow_2d_forward_valid_b2thw",
                "flow_2d_backward_valid_b2thw",
            }:
                raise NotImplementedError
            elif key in [
                "rgb_b3thw",
                "depth_b1thw",
                "depth_valid_b1thw",
                "instanceseg_b1thw",
                "dyn_mask_b1thw",
                "dyn_mask_valid_b1thw",
            ]:
                out[key] = sample[key].repeat(1, length, 1, 1)
            elif key in [
                "track_2d_traj_bn2t",
                "track_2d_depth_bn1t",
                "track_2d_vis_bn1t",
                "track_2d_valid_bn1t",
                "intrinsics_b44t",
            ]:
                out[key] = sample[key].repeat(1, 1, length)
            elif key in ["track_2d_pointquerries_bn3", "track_2d_pointlabels_bn"]:
                out[key] = sample[key]
            elif key == "extrinsics_b44t":
                out[key] = torch.eye(
                    4, dtype=sample["intrinsics_b44t"].dtype, device=sample["intrinsics_b44t"].device
                )[..., None].repeat(1, 1, length)
            elif key == "rel_pose_b6t":
                out[key] = torch.zeros(
                    6, length, dtype=sample["intrinsics_b44t"].dtype, device=sample["intrinsics_b44t"].device
                )
            else:
                raise NotImplementedError

        return out

    def resize(
        self,
        sample: Dict[str, torch.Tensor],
        resize_size: Tuple[int, int] = default_sample_size[1:],
        resize_mode: Dict[str, str] = {},
    ) -> Dict[str, torch.Tensor]:

        _, T, H, W = sample["rgb_b3thw"].shape
        resize_factor = (resize_size[0] / H, resize_size[1] / W)

        if (resize_factor[0] == 1.0) and (resize_factor[1] == 1.0):
            return sample

        # new_size = (T, int(H * resize_factor[0]), int(W * resize_factor[1]))
        new_size = (T, resize_size[0], resize_size[1])

        for key in sample.keys():
            if key in [
                "rgb_b3thw",
                "depth_b1thw",
                "depth_valid_b1thw",
                "instanceseg_b1thw",
                "flow_2d_backward_b2thw",
                "flow_2d_forward_b2thw",
                "flow_2d_backward_valid_b2thw",
                "flow_2d_forward_valid_b2thw",
                "dyn_mask_b1thw",
                "dyn_mask_valid_b1thw",
            ]:
                sample[key] = F.interpolate(sample[key][None], new_size, mode=resize_mode[key])[0]
                if key in ["flow_2d_backward_b2thw", "flow_2d_forward_b2thw"]:
                    sample[key][0] = sample[key][0] * resize_factor[1]  # u , multiply W
                    sample[key][1] = sample[key][1] * resize_factor[0]  # v , multiply H
            elif key in ["track_2d_traj_bn2t"]:
                sample[key][:, 0, :] = sample[key][:, 0, :] * resize_factor[1]  # x , multiply W
                sample[key][:, 1, :] = sample[key][:, 1, :] * resize_factor[0]  # y , multiply H
            elif key in [
                "track_2d_vis_bn1t",
                "track_2d_depth_bn1t",
                "track_2d_valid_bn1t",
                "extrinsics_b44t",
                "rel_pose_b6t",
            ]:
                continue
            elif key in ["intrinsics_b44t"]:
                sample[key][0, 0, :] = sample[key][0, 0, :] * resize_factor[1]  # x , multiply W
                sample[key][1, 1, :] = sample[key][1, 1, :] * resize_factor[0]  # y , multiply H
                sample[key][0, 2, :] = (sample[key][0, 2, :] + 0.5) * resize_factor[1] - 0.5  # x , multiply W
                sample[key][1, 2, :] = (sample[key][1, 2, :] + 0.5) * resize_factor[0] - 0.5  # y , multiply H
            else:
                print(f"key {key} not handled")
                raise NotImplementedError

        return sample

    def crop(
        self, sample: Dict[str, torch.Tensor], crop_size: Optional[Tuple[int, int, int]] = None
    ) -> Dict[str, torch.Tensor]:

        if crop_size is None:
            return sample

        _, T, H, W = sample["rgb_b3thw"].shape
        T_new, H_new, W_new = crop_size
        curr_shape = (T, H, W)
        diff_shape = [curr_shape[i] - crop_size[i] for i in range(3)]

        assert diff_shape[0] >= 0 and diff_shape[1] >= 0 and diff_shape[2] >= 0, print(
            f"Cropping Error: diff_shape {diff_shape}"
        )

        if diff_shape[0] == 0 and diff_shape[1] == 0 and diff_shape[2] == 0:
            return sample

        t0 = 0 if diff_shape[0] <= 0 else torch.randint(0, diff_shape[0], (1,))[0]
        if self.start_crop_time:
            t0 = 0  # TODO: Fix this hack. Specify that start should be 0 using an option
        if self.center_crop:
            i0 = 0 if diff_shape[1] <= 0 else int(diff_shape[1] * 0.5)
            j0 = 0 if diff_shape[2] <= 0 else int(diff_shape[2] * 0.5)
        else:
            i0 = 0 if diff_shape[1] <= 0 else torch.randint(0, diff_shape[1], (1,))[0]
            j0 = 0 if diff_shape[2] <= 0 else torch.randint(0, diff_shape[2], (1,))[0]

        # crop
        for key in sample.keys():
            if not torch.is_tensor(sample[key]):
                continue
            if key in [
                "rgb_b3thw",
                "flow_2d_backward_b2thw",
                "flow_2d_forward_b2thw",
                "depth_b1thw",
                "instanceseg_b1thw",
                "flow_2d_backward_valid_b2thw",
                "flow_2d_forward_valid_b2thw",
                "depth_valid_b1thw",
                "dyn_mask_b1thw",
                "dyn_mask_valid_b1thw",
            ]:
                sample[key] = sample[key][:, t0 : t0 + T_new, i0 : i0 + H_new, j0 : j0 + W_new]
            elif key in [
                "track_2d_traj_bn2t",
                "track_2d_vis_bn1t",
                "track_2d_depth_bn1t",
                "track_2d_valid_bn1t",
                "intrinsics_b44t",
                "extrinsics_b44t",
                "rel_pose_b6t",
            ]:
                sample[key] = sample[key][..., t0 : t0 + T_new]
            elif key in ["track_2d_pointlabels_bn", "track_2d_pointquerries_bn3"]:
                sample[key] = sample[key]
            else:
                print(f"key {key} not handled")
                raise NotImplementedError

        # remove all point querries that are outside and remove the correspoinding tracks
        # self.remove_queries_outside_bounds = False
        if "track_2d_pointquerries_bn3" in sample.keys() and self.remove_queries_outside_bounds:
            key = "track_2d_pointquerries_bn3"
            valid = torch.logical_and(sample[key][:, 0] > t0, sample[key][:, 0] < t0 + T_new)
            valid = torch.logical_and(
                valid, torch.logical_and(sample[key][:, 1] > j0, sample[key][:, 1] < j0 + W_new)
            )
            valid = torch.logical_and(
                valid, torch.logical_and(sample[key][:, 2] > i0, sample[key][:, 2] < i0 + H_new)
            )
            sample[key] = sample[key][valid, :]
            for key in sample.keys():
                if key in [
                    "track_2d_traj_bn2t",
                    "track_2d_vis_bn1t",
                    "track_2d_depth_bn1t",
                    "track_2d_valid_bn1t",
                    "track_2d_pointlabels_bn",
                ]:
                    sample[key] = sample[key][valid]

        # post-cropping handling
        if "track_2d_traj_bn2t" in sample.keys():
            sample["track_2d_traj_bn2t"][:, 0, :] = sample["track_2d_traj_bn2t"][:, 0, :] - j0
            sample["track_2d_traj_bn2t"][:, 1, :] = sample["track_2d_traj_bn2t"][:, 1, :] - i0

            sample["track_2d_vis_bn1t"][:, 0][sample["track_2d_traj_bn2t"][:, 0] >= (crop_size[2])] = False
            sample["track_2d_vis_bn1t"][:, 0][sample["track_2d_traj_bn2t"][:, 0] < 0] = False
            sample["track_2d_vis_bn1t"][:, 0][sample["track_2d_traj_bn2t"][:, 1] >= (crop_size[1])] = False
            sample["track_2d_vis_bn1t"][:, 0][sample["track_2d_traj_bn2t"][:, 1] < 0] = False

        if "intrinsics_b44t" in sample.keys():
            sample["intrinsics_b44t"][0, 2, :] = sample["intrinsics_b44t"][0, 2, :] - j0
            sample["intrinsics_b44t"][1, 2, :] = sample["intrinsics_b44t"][1, 2, :] - i0

        if "track_2d_pointquerries_bn3" in sample.keys():
            sample["track_2d_pointquerries_bn3"][:, 0] = sample["track_2d_pointquerries_bn3"][:, 0] - t0
            sample["track_2d_pointquerries_bn3"][:, 1] = sample["track_2d_pointquerries_bn3"][:, 1] - j0
            sample["track_2d_pointquerries_bn3"][:, 2] = sample["track_2d_pointquerries_bn3"][:, 2] - i0

        return sample

    def generate_point_qurries(self, traj: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:

        N, _, T = vis.shape
        vis_cumsum = torch.cumsum(vis.to(dtype=torch.int32), dim=-1)

        # add 0.5 so that temporal index are center of the frame duration
        traj_pts = torch.cat([torch.arange(T)[None, None, :].repeat(N, 1, 1) + 0.5, traj], dim=1)

        querry_pts = []
        for i in range(N):
            rand_num = torch.rand(1)
            # rand_num = 0 # TODO: HACK FOR QUERY FIRST SAMPLING
            sample_id = vis_cumsum[i, 0, :] == torch.round(rand_num * (vis_cumsum[i, 0, -1] - 1) + 1)
            sample_id = (sample_id).nonzero(as_tuple=False)[0, 0]
            assert vis[i, 0, sample_id], print("Something is wrong in sampling the traj queries")
            querry_pts.append(traj_pts[i, :, sample_id][None])

        querry_pts = torch.cat(querry_pts, dim=0)

        return querry_pts

    def sample_tracks(
        self, sample: Dict[str, torch.Tensor], sample_str: Dict[str, str], repeat_trajectories: bool = False
    ) -> Dict[str, torch.Tensor]:

        _, T, H, W = sample["rgb_b3thw"].shape
        txy_size = (T, W, H)

        if "track_2d_pointquerries_bn3" in sample.keys():
            assert "track_2d_pointlabels_bn" in sample.keys(), "need to pass point labels for %s}" % (
                sample_str["seq_name"]
            )
            assert "track_2d_valid_bn1t" in sample.keys(), "need to pass valid %s" % (sample_str["seq_name"])

        else:
            # Generate track point querries for the input video using grid sampling

            use_grid_sample = False
            if self.track_2d_querry_sampling_version is not None:
                use_grid_sample = True

            if use_grid_sample:
                # Generate grid points in the range [0, 1]
                grid_x, grid_y = torch.meshgrid(
                    torch.arange(0, 1, self.track_2d_querry_sampling_spacing),
                    torch.arange(0, 1, self.track_2d_querry_sampling_spacing),
                    indexing="xy",
                )
                # Sample in first frame
                dummy = torch.cat(
                    [torch.zeros_like(grid_x)[..., None], grid_x[..., None], grid_y[..., None]], dim=-1
                ).reshape(-1, 3)
                # Only sample points inside the instance seg mask
                if self.track_2d_querry_sampling_version == "uniform_over_seg":
                    valid_ids = []
                    # erode the instance seg a bit as instance seg can be noisy
                    instance_seg = sample["instanceseg_b1thw"][0, 0, :, :].clone()
                    kernel = torch.ones(3, 3)
                    instance_seg = erosion(instance_seg[None, None], kernel)[0, 0]

                    for n in range(dummy.shape[0]):
                        x_id = int(dummy[n, 1] * 224)
                        y_id = int(dummy[n, 2] * 224)
                        if instance_seg[y_id, x_id] > 0:
                            valid_ids.append(n)
                    if len(valid_ids) > 0:
                        dummy = dummy[valid_ids]

                self.track_2d_traj_per_sample = dummy.shape[0]

            # Generate dummy GT using self.track_2d_traj_per_sample
            sample["track_2d_traj_bn2t"] = torch.zeros((self.track_2d_traj_per_sample, 2, T)).to(
                dtype=torch.float32
            )
            sample["track_2d_vis_bn1t"] = torch.zeros((self.track_2d_traj_per_sample, 1, T)) > 0
            sample["track_2d_depth_bn1t"] = torch.ones((self.track_2d_traj_per_sample, 1, T))
            sample["track_2d_valid_bn1t"] = torch.zeros((self.track_2d_traj_per_sample, 1, T)) > 0

            # use grid sampling or random sampling
            if use_grid_sample:
                sample["track_2d_pointquerries_bn3"] = dummy.to(dtype=torch.float32)  # type: ignore
            else:
                sample["track_2d_pointquerries_bn3"] = torch.rand((self.track_2d_traj_per_sample, 3)).to(
                    dtype=torch.float32
                )

            # sample querries in the first frame
            sample["track_2d_pointquerries_bn3"][..., 0] = 0

            # scale the querries according to the input dimension
            for i in range(3):
                sample["track_2d_pointquerries_bn3"][..., i] = (
                    torch.round(sample["track_2d_pointquerries_bn3"][..., i] * (txy_size[i] - 1)) + 0.5
                )

            # set the point labels to 1
            sample["track_2d_pointlabels_bn"] = torch.ones((self.track_2d_traj_per_sample)).to(
                dtype=torch.float32
            )

        return sample

    def fix_track_valid_for_causal_estimation(
        self, sample: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        if "track_2d_valid_bn1t" not in sample.keys() or len(self.estimation_directions) == 2:
            return sample

        time_nt = 0.5 + torch.arange(sample["track_2d_valid_bn1t"].shape[-1]).repeat(
            sample["track_2d_valid_bn1t"].shape[-3], 1
        )
        query_time_nt = sample["track_2d_pointquerries_bn3"][:, 0][:, None].repeat(
            1, sample["track_2d_valid_bn1t"].shape[-1]
        )
        if self.estimation_directions[0] == 1:
            valid_nt = time_nt >= query_time_nt
        else:
            valid_nt = time_nt <= query_time_nt

        sample["track_2d_valid_bn1t"] = torch.logical_and(sample["track_2d_valid_bn1t"], valid_nt[:, None, :])

        return sample

    def post_process(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # fix track for causal estimation
        sample = self.fix_track_valid_for_causal_estimation(sample)
        return sample

    def get_dict_with_valid_vals(self, sample: L4PData) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        sample2 = asdict(sample)
        out = {}
        out_str = {}
        for key in sample2.keys():
            if sample2[key] is not None:
                if isinstance(sample2[key], str):
                    out_str[key] = sample2[key]
                else:
                    out[key] = sample2[key]

        if "intrinsics_b44t" not in out.keys():
            intrinsics_b44t = torch.eye(4)[:, :, None].repeat(1, 1, out["rgb_b3thw"].shape[-3])
            out["intrinsics_b44t"] = intrinsics_b44t

        return out, out_str

    def __getitem__(self, index):
        sample, sample_str = self.get_dict_with_valid_vals(self.getitem_helper(index))

        # mirror and pad the sample in temporal dimension
        ori_video_len = sample["rgb_b3thw"].shape[-3]
        T_curr = ori_video_len
        crop_size = self.crop_size
        multiple = self.length_multiply_of
        if crop_size is None:
            T_new = ceil(max(T_curr, self.default_sample_size[0]) / multiple) * multiple
            crop_size = (T_new,) + self.default_sample_size[1:]

        if T_curr == 1:
            sample = self.repeat_single_frame(sample, crop_size[0])
        else:
            while T_curr < crop_size[0]:
                sample = self.mirror_and_pad(sample)
                T_curr = sample["rgb_b3thw"].shape[-3]

        # perform spatial resizing
        if self.resize_size is not None:
            sample = self.resize(sample, self.resize_size, self.resize_mode)

        # perform spatio-temporal random cropping
        sample = self.crop(sample, crop_size)

        # perform track_2d sampling
        sample = self.sample_tracks(sample, sample_str, self.track_2d_repeat_traj)

        # final postprocessings and augmentations
        sample = self.post_process(sample)

        # get in proper form
        mean = self.input_mean[:, None, None, None]
        std = self.input_std[:, None, None, None]
        sample["rgb_mean_b3111"] = mean
        sample["rgb_std_b3111"] = std
        sample["rgb_b3thw"] = (sample["rgb_b3thw"] - mean) / std
        for key in sample.keys():
            if torch.is_tensor(sample[key]):
                sample[key] = sample[key].contiguous()
        sample.update(sample_str)
        sample["ori_video_len"] = ori_video_len  # type: ignore

        return sample
