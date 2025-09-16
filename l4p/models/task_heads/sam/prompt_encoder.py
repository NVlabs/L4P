# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the directory of this source file.

# The original code is modified to work with our video encoder output
# Original code url: https://github.com/facebookresearch/segment-anything/tree/main/segment_anything/modeling/prompt_encoder.py
# Modifications copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# License file located in: l4p/models/task_heads/sam/LICENSE

import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int, int],
        input_image_size: Tuple[int, int, int],
        num_point_embeddings: int = 2,
        prompt_using_features: bool = False,
        num_prompt_feature_embeddings: int = 2,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.
        Original SAM prompt encoder modified to work with video encoder for tracking task.

        Arguments:
          embed_dim (int): The prompts' embedding dimension, same as the video encoder embedding dimension (1408 for VideoMAE)
          image_embedding_size (tuple(int, int, int)): The spatial size of the video embedding as (8, 16, 16) for VideoMAE.
          input_image_size (tuple(int, int, int)): The spatial size of the input video (16, 224, 224).
          num_point_embeddings (int): The number of point embeddings to use, default is 2.
          prompt_using_features (bool): Whether to use track features as prompts, default is False.
          num_prompt_feature_embeddings (int): The number of track feature embeddings to use, default is 2.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        # 3D position embedding for video
        self.pe_layer = PositionEmbeddingRandom3D(embed_dim // 2)

        # Default is two point embeddings: to indicate if the input point query is to be used or ignored
        self.num_point_embeddings = num_point_embeddings
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)

        # functionality to track using track features as inputs
        # Default is two embeddings:
        # one to indicate the track feature is new and one to indicate the track feature is predicted by previous window
        self.prompt_using_features = prompt_using_features
        self.num_prompt_feature_embeddings = num_prompt_feature_embeddings
        if self.prompt_using_features:
            self.prompt_feature_embeddings = nn.ModuleList(
                [nn.Embedding(1, embed_dim) for i in range(self.num_prompt_feature_embeddings)]
            )

        # default from SAM head, ignore for now
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_t)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_features(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Embeds trackfeature prompts"""

        # initialize with 0
        feature_embeddings = torch.zeros_like(features)

        # label 0 indicates new feature, so we initialize with 0 id embedding
        # cast type to feature_embeddings.dtype
        feature_embeddings[labels == 0] = (
            features[labels == 0] + self.prompt_feature_embeddings[0].weight
        ).to(dtype=feature_embeddings.dtype)
        # label 1 indicates pred feature so we add 1 id embedding to this
        feature_embeddings[labels == 1] = (
            features[labels == 1] + self.prompt_feature_embeddings[1].weight
        ).to(dtype=feature_embeddings.dtype)
        return feature_embeddings

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts"""

        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 3), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)

        # default from SAM head, ignore for now
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight

        for i in range(self.num_point_embeddings):
            point_embedding[labels == i] += self.point_embeddings[i].weight

        return point_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        features: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:  # embed points
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if features is not None:  # embed track features
            prompt_feature, prompt_feature_labels = features
            feature_embeddings = self._embed_features(prompt_feature, prompt_feature_labels)
            sparse_embeddings = torch.cat([sparse_embeddings, feature_embeddings], dim=1)

        return sparse_embeddings


class PositionEmbeddingRandom3D(nn.Module):
    """
    3D Positional encoding for video using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^3 cubeand have d_1 x ... x d_n x 3 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        t, h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((t, h, w), device=device, dtype=torch.float32)

        t_embed = grid.cumsum(dim=0) - 0.5
        y_embed = grid.cumsum(dim=1) - 0.5
        x_embed = grid.cumsum(dim=2) - 0.5
        t_embed = t_embed / t
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([t_embed, x_embed, y_embed], dim=-1))
        return pe.permute(3, 0, 1, 2)  # C x T x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Positionally encode points that are not normalized to [0,1]. image size : (T,H,W)
        Position encoding for txy
        """
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[0]  # t dimension
        coords[:, :, 1] = coords[:, :, 1] / image_size[2]  # x dimension
        coords[:, :, 2] = coords[:, :, 2] / image_size[1]  # y dimension
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
