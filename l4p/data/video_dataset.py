# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import torch
import einops
import mediapy as media
from PIL import Image
from typing import Tuple, List, Dict
from l4p.data.l4p_dataset_mini import L4PDataset, L4PData, ESTIMATION_DIRECTIONS
import torchvision.transforms.functional as F


class VideoDataset(L4PDataset):
    def __init__(
        self,
        video_paths: List[str],
        dataset_type: str = "video",
        max_frames: int = 192,
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
        """Video Dataset to load any generic video.

        Args:
            video_paths (List[str]): List of video paths
            dataset_type (str, optional): Give it a name. Defaults to "video".
            max_frames (int, optional): Maximum number of frames to load. Defaults to 192.
            stride (int, optional): Stride to load the video. Defaults to 1.
            crop_size (None | Tuple[int, int, int], optional): Crop to size (T, H, W). Defaults to None.
            resize_size (Tuple[int, int], optional): Resize the spatial resolution to (H,W) before cropping. Defaults to (224, 224).
            center_crop (bool, optional): Center or random crop. Defaults to True.
            start_crop_time (bool, optional): Start cropping at time 0 or random. Defaults to True.
            estimation_directions (List[ESTIMATION_DIRECTIONS], optional): Estimation direction for tracking. Defaults to [1].
            resize_mode (Dict[str, str], optional): Resize mode for each buffer. Defaults to {"rgb_b3thw": "trilinear"}.
            track_2d_querry_sampling_spacing (float, optional): Query poitns are sampled at a spacing of 0.02 for normalized image size [0,1]. Defaults to 0.02.
        """
        super(VideoDataset, self).__init__(
            crop_size=crop_size,
            center_crop=center_crop,
            start_crop_time=start_crop_time,
            estimation_directions=estimation_directions,
            resize_mode=resize_mode,
            resize_size=resize_size,
            track_2d_querry_sampling_version="uniform",
            track_2d_querry_sampling_spacing=track_2d_querry_sampling_spacing,
        )

        self.video_paths = video_paths
        self.max_frames = max_frames
        self.stride = stride
        self.dataset_type = dataset_type
        self.resize_size = resize_size
        self.len = len(self.video_paths)
        print(f"Found {self.len} in {self.__class__.__name__}")

    def __len__(self):
        return self.len

    def getitem_helper(self, index: int) -> L4PData:

        rgbs = []
        instances = []
        video_name = os.path.basename(self.video_paths[index])

        antialias_resize = True
        with media.VideoReader(self.video_paths[index]) as reader:
            print(f"Video has {reader.num_images} images with shape={reader.shape} at {reader.fps} fps")
            if reader.num_images > self.max_frames:
                print(
                    f"Trimming video to {self.max_frames} frames.",
                    "This is the limitation of current implementation.",
                )
            count = 0
            for rgb in reader:
                # rgb = media.resize_image(rgb, self.resize_size)
                rgb = Image.fromarray(rgb)
                if antialias_resize:
                    # convert to PIL Image
                    full_size = rgb.size
                    rgb = rgb.resize(self.resize_size, resample=Image.Resampling.BILINEAR)
                    rgb = rgb.resize(full_size, resample=Image.Resampling.BILINEAR)
                rgb = F.to_tensor(rgb)[:3][:, None]
                rgbs.append(rgb)
                instance = torch.zeros_like(rgb[:1])
                instances.append(instance)

                count += 1
                if count == self.max_frames - 1:
                    break

        rgb_b3thw = torch.cat(rgbs, dim=1)
        instance_b1thw = (torch.mean(torch.cat(instances, dim=1), dim=0, keepdim=True) > 0).to(
            dtype=torch.float32
        )

        rgb_b3thw = rgb_b3thw[:, :: self.stride]
        instance_b1thw = instance_b1thw[:, :: self.stride]

        # generate sample for L4PData
        sample = {}

        # pass dummy intrinsics
        _, T, H, W = rgb_b3thw.shape
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
            seq_name=str(video_name),
        )

        return sample
