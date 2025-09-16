# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the directory of this source file.

# The original code is modified to work with our video encoder output
# Original code url: https://github.com/facebookresearch/segment-anything/tree/main/segment_anything/modeling/mask_decoder.py
# Modifications copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# License file located in: l4p/models/task_heads/sam/LICENSE

from typing import List, Tuple, Type, Dict
import torch
from torch import nn
from torch.nn import functional as F


class MaskDecoder(nn.Module):
    """
    Original SAM mask decoder modified to work with video encoder output and for the tracking task.
    Given prompt embeddings (points and track features), outputs masks and processed video tokens.
    """

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
        num_mask_tokens: int = 1,
        decoding_out_dim_factor: int = 8
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          activation (nn.Module): the type of activation to use when upscaling masks
          num_mask_tokens (int): the number of mask tokens to use, default is 3 (2D track, visibility, depth)
          decode_mask (bool): whether to decode the mask, default is True
          decoding_out_dim_factor (int): the factor to decode the mask out dimension
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.decoding_out_dim_factor = decoding_out_dim_factor

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_mask_tokens  # default is 3 (2D track, visibility, depth)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        decode_mask_dims = [
            min(2 * transformer_dim // decoding_out_dim_factor, transformer_dim),
            transformer_dim // decoding_out_dim_factor,
        ]

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose3d(transformer_dim, decode_mask_dims[0], kernel_size=2, stride=2),
            LayerNorm3d(decode_mask_dims[0]),
            activation(),
            nn.ConvTranspose3d(
                decode_mask_dims[0], decode_mask_dims[1], kernel_size=(1, 2, 2), stride=(1, 2, 2)
            ),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, decode_mask_dims[1], 3)
                for i in range(self.num_mask_tokens)
            ]
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor | None, Dict[str, torch.Tensor]]:
        """
        Predicts video masks and processed video tokens given input and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the video encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and track features

        Returns:
          out (torch.Tensor): batched predicted masks
          sam_processed_features (Dict[str, torch.Tensor]): processed video and i/o tokens
        """

        out, sam_processed_features = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
        )

        return out, sam_processed_features

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor | None, Dict[str, torch.Tensor]]:
        """Predicts masks. See 'forward' for more details."""

        # Concatenate output tokens
        output_tokens = torch.cat([self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        # Add sparse tokens after the masked tokens
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = image_embeddings[0]  # 1 n p c -> n p c
        if src.shape[0] == 1:  # 1 p c -> n p c
            src = torch.repeat_interleave(src, tokens.shape[0], dim=0)
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, t, h, w = pos_src.shape
        pos_src = pos_src.flatten(2).transpose(1, 2)

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)

        # Copy the tokens for output
        sam_processed_features = {"io_features": hs.clone(), "enc_features": src.clone()}
        mask_tokens_out = hs

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [1, 1, 176]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, t, h, w)  # [1, 1408, 8, 16, 16]
        upscaled_embedding = self.output_upscaling(src)  # [1, 176, 16, 64, 64]
        b, c, t, h, w = upscaled_embedding.shape
        out = (hyper_in @ upscaled_embedding.view(b, c, t * h * w)).view(b, -1, t, h, w)  # [1, 1, 16, 64, 64]

        return out, sam_processed_features


# Original SAM LayerNorm2D code is modified for 3D input
class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
