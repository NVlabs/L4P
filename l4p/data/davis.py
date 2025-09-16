# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import glob
import torch
import einops
import numpy as np
import mediapy as media
from PIL import Image
from typing import Tuple, List, Dict
from l4p.data.l4p_dataset_mini import L4PDataset, L4PData, ESTIMATION_DIRECTIONS
import torchvision.transforms.functional as F


def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Resize a video to output_size."""
    return media.resize_video(video, output_size)


class DavisDataset(L4PDataset):
    def __init__(
        self,
        data_root: str,
        dataset_type: str = "davis",
        stride: int = 1,
        ### Base class args below ###
        crop_size: None | Tuple[int, int, int] = None,
        resize_size: Tuple[int, int] = (224, 224),
        center_crop: bool = True,
        start_crop_time: bool = True,
        estimation_directions: List[ESTIMATION_DIRECTIONS] = [1],
        resize_mode: Dict[str, str] = {"rgb_b3thw": "trilinear"},
        track_2d_querry_sampling_spacing: float = 0.02,
    ):
        """Davis Dataset

        Args:
            data_root (str): Data root path
            dataset_type (str, optional): Give it a name. Defaults to "davis".
            stride (int, optional): Stride to load the video. Defaults to 1.
            crop_size (None | Tuple[int, int, int], optional): Crop to size (T, H, W). Defaults to None.
            resize_size (Tuple[int, int], optional): Resize the spatial resolution to (H,W) before cropping. Defaults to (224, 224).
            center_crop (bool, optional): Center or random crop. Defaults to True.
            start_crop_time (bool, optional): Start cropping at time 0 or random. Defaults to True.
            estimation_directions (List[ESTIMATION_DIRECTIONS], optional): Estimation direction for tracking. Defaults to [1].
            resize_mode (Dict[str, str], optional): Resize mode for each buffer. Defaults to {"rgb_b3thw": "trilinear"}.
            track_2d_querry_sampling_spacing (float, optional): Query poitns are sampled at a spacing of 0.02 for normalized image size [0,1]. Defaults to 0.02.
        """
        super(DavisDataset, self).__init__(
            crop_size=crop_size,
            center_crop=center_crop,
            start_crop_time=start_crop_time,
            estimation_directions=estimation_directions,
            resize_mode=resize_mode,
            resize_size=resize_size,
            track_2d_querry_sampling_version="uniform_over_seg",
            track_2d_querry_sampling_spacing=track_2d_querry_sampling_spacing,
        )
        print(f"Loading {dataset_type}")
        self.data_root = data_root
        self.dataset_type = dataset_type
        self.resize_size = resize_size
        self.stride = stride
        self.scene_list = sorted(glob.glob(os.path.join(self.data_root, "JPEGImages/480p/*")))
        self.len = len(self.scene_list)
        print("found %d unique videos in %s" % (len(self.scene_list), data_root))

    def __len__(self):
        return self.len

    def getitem_helper(self, index: int) -> L4PData:

        rgbs = []
        instances = []
        antialias_resize = True
        video_name = os.path.basename(self.scene_list[index])
        seq_name = str(video_name)
        T = len(glob.glob(os.path.join(self.scene_list[index], "*.jpg")))

        for i in range(0, T, self.stride):
            rgb_path = os.path.join(self.scene_list[index], "%05d.jpg" % i)
            rgb = Image.open(rgb_path)
            if antialias_resize:
                full_size = rgb.size
                rgb = rgb.resize(self.resize_size, resample=Image.Resampling.BILINEAR)
                rgb = rgb.resize(full_size, resample=Image.Resampling.BILINEAR)

            rgb = F.to_tensor(rgb)[:3][:, None]
            rgbs.append(rgb)

            # load instance mask, used to sample query points on top of the instances
            instance_path = rgb_path.replace("JPEGImages", "Annotations").replace("jpg", "png")
            if os.path.isfile(instance_path):
                instance = Image.open(instance_path)
                if antialias_resize:
                    full_size = instance.size
                    instance = instance.resize(self.resize_size, resample=Image.Resampling.BILINEAR)
                    instance = instance.resize(full_size, resample=Image.Resampling.BILINEAR)

                instance = F.to_tensor(instance)[:1][:, None]
            else:
                instance = torch.zeros_like(rgb[:1])
            instances.append(instance)

        rgb_b3thw = torch.cat(rgbs, dim=1)
        instance_b1thw = (torch.mean(torch.cat(instances, dim=1), dim=0, keepdim=True) > 0).to(
            dtype=torch.float32
        )

        # generate sample for L4PData
        sample = {}

        # pass dummy intrinsics
        _, _, H, W = rgb_b3thw.shape
        fx = min(H, W)
        fy = min(H, W)
        cx = W / 2
        cy = H / 2
        intrinsics_b44 = torch.Tensor(
            [
                [fx, 0, cx, 0],
                [0, fy, cy, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        intrinsics_b44t = einops.repeat(intrinsics_b44, "m n -> m n k", k=rgb_b3thw.shape[-3]).clone()

        # pass to the base class
        sample = L4PData(
            rgb_b3thw=rgb_b3thw,
            intrinsics_b44t=intrinsics_b44t,
            instanceseg_b1thw=instance_b1thw,
            seq_name=seq_name,
        )

        return sample
