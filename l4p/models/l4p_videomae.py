# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
from functools import partial
from typing import Optional, List, Dict, Any, Tuple

from l4p.models.VideoMAEv2.models.modeling_pretrain import PretrainVisionTransformerEncoder
from l4p.models.task_heads.dense_heads import joint_windowed_estimation
from l4p.utils.geometry_utils import normalize_intrinsics
from l4p.models.blocks import PluckerCameraEmbedding


class VideoMAEEncoder(PretrainVisionTransformerEncoder):
    """
    Wrapper to use VideoMAE encoder for L4P.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0,
        norm_layer=torch.nn.LayerNorm,
        init_values=None,
        tubelet_size=2,
        use_learnable_pos_emb=False,
        with_cp=False,
        all_frames=16,
        cos_attn=False,
        cam_emb_placed_at=None,  # None | 'input' | 'output'
        cam_emb_type="add",  # 'add' | 'concat'
    ):
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            num_classes,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            init_values,
            tubelet_size,
            use_learnable_pos_emb,
            with_cp,
            all_frames,
            cos_attn,
        )
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        self.cam_emb_placed_at = cam_emb_placed_at
        if cam_emb_placed_at is None:
            self.cam_emb = None
        else:
            self.cam_emb = PluckerCameraEmbedding(embed_dim=embed_dim, embed_type=cam_emb_type)

        return

    def forward(self, x, intrinsics_b44t=None, extrinsics_b44t=None):
        """
        Forward pass for the VideoMAE encoder.

        Args:
            x (torch.Tensor): Input video frames (B, 3, T, H, W).
            intrinsics_b44t (torch.Tensor, optional): Normalized Intrinsics (B, 4, 4, T). Defaults to None.
            extrinsics_b44t (torch.Tensor, optional): Extrinsics (B, 4, 4, T). Defaults to None.

        Returns:
            List[torch.Tensor]: List of features for each block. Features are of shape (B, num_tokens, embed_dim)
        """
        B, _, T, H, W = x.shape
        emb_thw = (
            T // self.tubelet_size,
            H // self.patch_size,
            W // self.patch_size,
        )

        x = self.patch_embed(x)

        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        if self.cam_emb and self.cam_emb_placed_at == "input":
            x = self.cam_emb(x, emb_thw, intrinsics_b44t, extrinsics_b44t)  # x -- b x l x c

        x_vis = x

        features_list = []
        features_list.append(x_vis)

        for blk in self.blocks:
            x_vis = blk(features_list[-1])
            features_list.append(x_vis)

        features_list[-1] = self.head(self.norm(features_list[-1]))

        if self.cam_emb and self.cam_emb_placed_at == "output":
            features_list = [
                self.cam_emb(feat, emb_thw, intrinsics_b44t, extrinsics_b44t) for feat in features_list
            ]

        return features_list


class L4P_VideoMAE(torch.nn.Module):
    """
    Main L4P Model with VideoMAE encoder and task heads.
    """

    def __init__(
        self,
        task_heads: torch.nn.ModuleDict,
        video_encoder_ckpt_path: Optional[str] = None,
        window_size: Tuple[int, int, int] = (16, 224, 224),
        window_stride_T: int = 8,
        freeze_video_encoder: bool = False,
        freeze_heads: Optional[List[str]] = None,
        unfreeze_blocks: Optional[List[int]] = None,
        always_use_windowed_version: bool = False,
        joint_alignment: bool = False,
        cam_emb_placed_at_enc: Optional[str] = None,
        cam_emb_type: str = "add",
    ) -> None:
        """
        Initialize the L4P_VideoMAE model.

        Args:
            task_heads (torch.nn.ModuleDict): Task heads to use for the model.
            video_encoder_ckpt_path (Optional[str], optional): Opitonal path to the video encoder checkpoint for weights initialization.
            window_size (Tuple[int, int, int], optional): VideoMAE window size (T, H, W). Defaults to (16, 224, 224).
            window_stride_T (int, optional): Widow stride for sliding window inference. Defaults to T/2 = 8.
            freeze_video_encoder (bool, optional): Freeze video encoder parameters. Defaults to False.
            freeze_heads (Optional[List[str]], optional): List of task heads to freeze. Defaults to None.
            unfreeze_blocks (Optional[List[int]], optional): List of blocks to unfreeze in the video encoder. Defaults to None.
            always_use_windowed_version (bool, optional): Force sliding window inference. Defaults to False.
            joint_alignment (bool, optional): Perform joint alignment for depth and camray tasks. Defaults to False.
            cam_emb_placed_at_enc (Optional[str], optional): Optional functionality to pass cameara embeddings. Defaults to None.
            cam_emb_type (str, optional): How to pass the camera embeddings. Defaults to "add".
        """
        super().__init__()

        # Initialize the VideoMAE encoder
        self.video_encoder = VideoMAEEncoder(
            img_size=224,
            patch_size=14,
            in_chans=3,
            num_classes=0,
            embed_dim=1408,
            depth=40,
            num_heads=16,
            mlp_ratio=48 / 11,  # type: ignore
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0,
            attn_drop_rate=0,
            drop_path_rate=0,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),  # type: ignore
            init_values=0.0,
            tubelet_size=2,
            use_learnable_pos_emb=False,
            with_cp=False,
            all_frames=16,
            cos_attn=False,
            cam_emb_placed_at=cam_emb_placed_at_enc,
            cam_emb_type=cam_emb_type,
        )
        if video_encoder_ckpt_path is not None:
            print(f"Loading video model: {video_encoder_ckpt_path}")
            ckpt = torch.load(video_encoder_ckpt_path, map_location="cpu", weights_only=True)
            self.video_encoder.load_state_dict(ckpt, strict=False)
            print("Successfully loaded video model")

        self.task_heads = task_heads
        self.window_size = window_size
        self.window_stride_T = window_stride_T
        self.always_use_windowed_version = always_use_windowed_version
        self.joint_alignment = joint_alignment

        if freeze_video_encoder:
            print("Freezing video encoder parameters")
            for params in self.video_encoder.parameters():
                params.requires_grad = False
            if unfreeze_blocks is not None:
                print("Unfreezing final head and norm")
                for params in self.video_encoder.head.parameters():
                    params.requires_grad = True
                for params in self.video_encoder.norm.parameters():
                    params.requires_grad = True
                print("Unfreezing VIT blocks:", unfreeze_blocks)
                for i in unfreeze_blocks:
                    for params in self.video_encoder.blocks[i].parameters():
                        params.requires_grad = True

        if freeze_heads is not None:
            print("Freezing task heads", freeze_heads)
            for task in freeze_heads:
                for params in self.task_heads[task].parameters():
                    params.requires_grad = False

        return

    def encode_features(self, data: Dict[str, Any]):
        """Geneartes video encoder features for a single window."""
        h, w = data["rgb_b3thw"].shape[-2:]
        intrinsics = (
            normalize_intrinsics(data["intrinsics_b44t"], h, w) if "intrinsics_b44t" in data else None
        )
        return self.video_encoder(
            data["rgb_b3thw"],
            intrinsics,
            data["extrinsics_b44t"] if "extrinsics_b44t" in data else None,
        )

    def forward_single_window(self, data: Dict[str, Any], tasks: List[str]) -> Dict[str, Any]:
        """Generates video encoder features for a single window and estimates output for each task."""
        h, w = data["rgb_b3thw"].shape[-2:]
        # normalize intrinsics if provided
        intrinsics = (
            normalize_intrinsics(data["intrinsics_b44t"], h, w) if "intrinsics_b44t" in data else None
        )
        # generate video encoder features
        enc_features_bpc_list = self.video_encoder(
            data["rgb_b3thw"],
            intrinsics,
            data["extrinsics_b44t"] if "extrinsics_b44t" in data else None,
        )
        # for each task, decode the video encoder features to estimate output
        out = {}
        out["enc_features_bpc_list"] = enc_features_bpc_list
        for task in tasks:
            task_out = self.task_heads[task](enc_features_bpc_list=enc_features_bpc_list, **data)
            out.update(task_out)

        return out

    def forward(self, data: Dict[str, Any], tasks: List[str]) -> Dict[str, Any]:
        """Main forward pass for both single and multi-window inference."""

        B, _, T, H, W = data["rgb_b3thw"].shape
        assert H == self.window_size[1] and W == self.window_size[2], "Supports only fixed spatial size"

        # Single window inference
        if (not self.always_use_windowed_version) and (T == self.window_size[0]):
            return self.forward_single_window(data, tasks)

        # Multi-window inference
        assert (
            T % self.window_stride_T == 0
        ), "Temporal window needs to be a multiple of window stride, for now!"
        time_strides = torch.arange(0, T - self.window_size[0] + 1, self.window_stride_T)

        intrinsics = (
            normalize_intrinsics(data["intrinsics_b44t"], H, W) if "intrinsics_b44t" in data else None
        )

        # First generate the video encoder features for all the windows and create a 2D list of features
        enc_features_bpc_2dlist = []
        for curr_start_id in time_strides:
            # Get the current window features from video encoder
            enc_features_bpc_list = self.video_encoder(
                data["rgb_b3thw"][:, :, curr_start_id : curr_start_id + self.window_size[0], :, :],
                (
                    intrinsics[:, :, :, curr_start_id : curr_start_id + self.window_size[0]]
                    if intrinsics is not None
                    else None
                ),
                (
                    data["extrinsics_b44t"][:, :, :, curr_start_id : curr_start_id + self.window_size[0]]
                    if "extrinsics_b44t" in data
                    else None
                ),
            )
            enc_features_bpc_2dlist.append(enc_features_bpc_list)

        out = {}
        out["enc_features_bpc_2dlist"] = enc_features_bpc_2dlist

        # For each task generate the sliding window inference using task-specific alignment approach
        joint_alignment_possible = "depth" in tasks and "camray" in tasks
        if self.joint_alignment and joint_alignment_possible:
            # other tasks are still done separately
            for task in ["track_2d", "dyn_mask", "flow_2d_backward"]:
                if task in tasks:
                    task_out = self.task_heads[task].forward_windowed(
                        enc_features_bpc_2dlist=enc_features_bpc_2dlist, time_strides=time_strides, **data
                    )
                    out.update(task_out)
            # joint alignment for depth and camray
            assert (
                "depth" in tasks and "camray" in tasks
            ), "Depth and camray must be present for joint alignment"
            tasks_out = joint_windowed_estimation(
                ["depth", "camray"],
                self.task_heads,
                enc_features_bpc_2dlist=enc_features_bpc_2dlist,
                time_strides=time_strides,
                **data,
            )
            out.update(tasks_out)
        else:
            if self.joint_alignment:
                print("Joint alignment is not possible as depth or camray tasks are not present")
            # every task has its own alignment approach
            for task in tasks:
                task_out = self.task_heads[task].forward_windowed(
                    enc_features_bpc_2dlist=enc_features_bpc_2dlist, time_strides=time_strides, **data
                )
                out.update(task_out)

        return out
