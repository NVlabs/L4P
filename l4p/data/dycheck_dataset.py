# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import glob
from typing import Dict, Tuple, Optional, List
import PIL.Image
from PIL.ImageOps import exif_transpose
import torch
import torchvision.transforms.functional as F
from l4p.data.l4p_dataset_mini import L4PData, L4PDataset, ESTIMATION_DIRECTIONS


class DycheckDataset(L4PDataset):
    def __init__(
        self,
        data_root: str,
        dataset_type: str = "dycheck",
        stride: int = 1,
        ### Base class args below ###
        crop_size: Optional[Tuple[int, int, int]] = None,
        resize_size: Optional[Tuple[int, int] | int] = (224, 224),
        center_crop: bool = True,
        start_crop_time: bool = True,
        estimation_directions: List[ESTIMATION_DIRECTIONS] = [1],
        resize_mode: Dict[str, str] = {"depth_b1thw": "trilinear"},
        track_2d_querry_sampling_spacing: float = 0.02,
    ):
        """Dycheck Dataset to load input RGB and GT intrinsics.

        Args:
            data_root (str): Data root path
            dataset_type (str, optional): Give it a name. Defaults to "dycheck".
            stride (int, optional): Stride to load the video. Defaults to 1.
            crop_size (None | Tuple[int, int, int], optional): Crop to size (T, H, W). Defaults to None.
            resize_size (Tuple[int, int], optional): Resize the spatial resolution to (H,W) before cropping. Defaults to (224, 224).
            center_crop (bool, optional): Center or random crop. Defaults to True.
            start_crop_time (bool, optional): Start cropping at time 0 or random. Defaults to True.
            estimation_directions (List[ESTIMATION_DIRECTIONS], optional): Estimation direction for tracking. Defaults to [1].
            resize_mode (Dict[str, str], optional): Resize mode for each buffer. Defaults to {"rgb_b3thw": "trilinear"}.
            track_2d_querry_sampling_spacing (float, optional): Query poitns are sampled at a spacing of 0.02 for normalized image size [0,1]. Defaults to 0.02.
        """
        super().__init__(
            crop_size=crop_size,
            center_crop=center_crop,
            start_crop_time=start_crop_time,
            resize_size=resize_size,
            resize_mode=resize_mode,
            estimation_directions=estimation_directions,
            track_2d_querry_sampling_version="uniform",
            track_2d_querry_sampling_spacing=track_2d_querry_sampling_spacing,
        )
        self.data_root = data_root
        self.stride = stride
        self.dataset_type = dataset_type
        self.seq_list = sorted(glob.glob(os.path.join(data_root, "*")))
        print("Dataset size: ", len(self.seq_list))

        return

    def __len__(self):
        return len(self.seq_list)

    def getitem_helper(self, index: int) -> L4PData:
        dir_path = self.seq_list[index]
        seq = dir_path.split("/")[-1]
        seq_name = f"Dycheck_{seq}"

        dir_path = os.path.join(self.data_root, seq)

        img_list = sorted(glob.glob(os.path.join(dir_path, "dense", "images", "*.png")))[:: self.stride]
        # read rgb images using PIL and generate rgb_b3thw tensor
        rgbs = []
        for img_path in img_list:
            img = exif_transpose(PIL.Image.open(img_path)).convert("RGB")  # type: ignore
            rgbs.append(F.to_tensor(img)[:3][:, None])
        rgb_b3thw = torch.cat(rgbs, dim=1)

        # Get GT intrinsics
        calibration_file = os.path.join(dir_path, "calibration.txt")
        with open(calibration_file, "r") as f:
            lines = f.readlines()
        fx = float(lines[0].split(" ")[0])
        fy = float(lines[0].split(" ")[1])
        cx = float(lines[0].split(" ")[2])
        cy = float(lines[0].split(" ")[3])

        intrinsics_b44t = torch.eye(4).to(dtype=torch.float32)
        intrinsics_b44t[0, 0] = fx
        intrinsics_b44t[1, 1] = fy
        intrinsics_b44t[0, 2] = cx
        intrinsics_b44t[1, 2] = cy
        intrinsics_b44t = intrinsics_b44t.unsqueeze(-1).repeat(1, 1, rgb_b3thw.shape[1])

        # dummy extrinsics
        extrinsics_b44t = torch.eye(4).to(dtype=torch.float32)
        extrinsics_b44t = extrinsics_b44t.unsqueeze(-1).repeat(1, 1, rgb_b3thw.shape[1])

        sample = L4PData(
            rgb_b3thw=rgb_b3thw,
            intrinsics_b44t=intrinsics_b44t,
            extrinsics_b44t=extrinsics_b44t,
            seq_name=seq_name,
        )

        return sample
