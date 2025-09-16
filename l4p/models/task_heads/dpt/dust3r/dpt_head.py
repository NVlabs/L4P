# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dpt head implementation for DUST3R
# Downstream heads assume inputs of size B x N x C (where N is the number of tokens) ;
# or if it takes as input the output at every layer, the attribute return_all_layers should be set to True
# the forward function also takes as input a dictionnary img_info with key "height" and "width"
# for PixelwiseTask, the output will be of dimension B x num_channels x H x W
# --------------------------------------------------------

# The original code is modified to work with our video encoder output
# Original code url: https://github.com/naver/dust3r/blob/main/dust3r/heads/dpt_head.py
# Modifications copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# License file located in: l4p/models/task_heads/dpt/dust3r/LICENSE


from einops import rearrange
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from l4p.models.task_heads.dpt.croco.dpt_block import DPTOutputAdapter  # noqa


class DPTOutputAdapter_fix(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weigths, and fix forward for dust3r
    """

    def init(self, dim_tokens_enc=1408):
        super().init(dim_tokens_enc)
        # these are duplicated weights
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

    def forward(self, encoder_tokens: List[torch.Tensor], image_size=None):
        assert self.dim_tokens_enc is not None, "Need to call init(dim_tokens_enc) function first"
        image_size = self.image_size if image_size is None else image_size
        T, H, W = image_size  # type: ignore
        # Number of patches in height and width
        N_T = T // (self.stride_level * self.P_T)
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)
        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]

        # Extract only task-relevant tokens and ignore global tokens.
        layers = [self.adapt_tokens(layer) for layer in layers]

        # Reshape tokens to spatial representation
        # [[1, 1408, 8, 16, 16], [1, 1408, 8, 16, 16], [1, 1408, 8, 16, 16], [1, 1408, 8, 16, 16]]
        layers = [
            rearrange(layer, "b (nt nh nw) c -> b c nt nh nw", nt=N_T, nh=N_H, nw=N_W).contiguous()
            for layer in layers
        ]

        # [[1, 256, 16, 64, 64], [1, 512, 16, 32, 32], [1, 1024, 8, 16, 16], [1, 1024, 4, 8, 8]]
        layers = [self.act_postprocess[idx](layer) for idx, layer in enumerate(layers)]

        # Project layers to chosen feature dim
        # [[1, 256, 16, 64, 64], [1, 256, 16, 32, 32], [1, 256, 8, 16, 16], [1, 256, 4, 8, 8]]
        layers = [self.scratch.layer_rn[idx](layer) for idx, layer in enumerate(layers)]

        # Fuse layers using refinement stages
        path_4 = self.scratch.refinenet4(layers[3])[
            :, :, : layers[2].shape[2], : layers[2].shape[3]
        ]  # [1, 256, 8, 16, 16])
        path_3 = self.scratch.refinenet3(path_4, layers[2])  # [1, 256, 16, 32, 32]
        path_2 = self.scratch.refinenet2(path_3, layers[1])  # [1, 256, 16, 64, 64]
        path_1 = self.scratch.refinenet1(path_2, layers[0])  # [1, 256, 16, 128, 128]

        # Output head
        out = self.head1(path_1)  # [1, 128, 16, 128, 128]
        output_size = image_size if self.output_size is None else self.output_size
        if out.shape[-3:] != output_size:
            out = F.interpolate(
                out, size=output_size, mode="trilinear", align_corners=True
            )  # [1, 128, 16, 224, 224]
        out = self.head2(out)  # [1, 2, 16, 224, 224]

        return out


class PixelwiseTaskWithDPT(nn.Module):
    """DPT module for dust3r, can return 3D points + confidence for all pixels"""

    def __init__(
        self,
        *,
        n_cls_token=0,
        hooks_idx=None,
        dim_tokens=None,
        output_width_ratio=1,
        num_channels=1,
        **kwargs,
    ):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers

        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(output_width_ratio=output_width_ratio, num_channels=num_channels, **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)  # type: ignore
        dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, img_info: Tuple[int, int, int] = (16, 224, 224)):
        out = self.dpt(x, image_size=(img_info[0], img_info[1], img_info[2]))
        return out
