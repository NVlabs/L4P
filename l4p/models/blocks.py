# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
import torch.nn as nn

from l4p.utils.geometry_utils import get_rays_plucker


class PluckerCameraEmbedding(nn.Module):
    def __init__(self, embed_dim, embed_type) -> None:
        super().__init__()
        proj_in_dim = 6
        if embed_type == "concat":
            proj_in_dim += embed_dim
        else:
            assert embed_type == "add"

        self.cam_emb_proj = torch.nn.Linear(proj_in_dim, embed_dim)
        self.embed_dim = embed_dim
        self.embed_type = embed_type

    def forward(self, feat_blc, emb_thw, intrinsics_b44t, extrinsics_b44t):
        # expects normalized intrinsics
        # extrinsics is cam_T_world (pose is world_T_cam)

        B, _, _, T = intrinsics_b44t.shape
        h, w = emb_thw[1:]

        camray_b6thw, _ = get_rays_plucker(intrinsics_b44t, extrinsics_b44t, (h, w), normalize_dist=False)

        # subsample -- TODO: proper interpolation of poses
        plucker = (
            nn.functional.interpolate(
                camray_b6thw.permute(0, 3, 4, 1, 2).reshape(B, -1, T), size=emb_thw[0], mode="linear"
            )
            .reshape(B, h, w, 6, emb_thw[0])
            .permute(0, 4, 1, 2, 3)
            .reshape(B, -1, 6)
        )

        # project to output
        if self.embed_type == "concat":
            plucker_emb = torch.cat([feat_blc, plucker], dim=-1)
            plucker_emb = self.cam_emb_proj(plucker_emb)
        elif self.embed_type == "add":
            plucker_emb = self.cam_emb_proj(plucker)
        feat_blc = feat_blc + plucker_emb

        return feat_blc
