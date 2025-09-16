# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
import einops
import math
from typing import Tuple, List, Dict, Optional, Literal

from l4p.models.task_heads.sam.prompt_encoder import PromptEncoder
from l4p.models.task_heads.sam.mask_decoder import MaskDecoder
from l4p.models.task_heads.sam.transformer import TwoWayTransformer

from l4p.utils.misc import apply_fn


class VideoMAETrack2DSamHead(torch.nn.Module):
    def __init__(
        self,
        task_name: str = "track_2d",
        prompt_embed_dim: int = 1408,
        image_size: Tuple[int, int, int] = (16, 224, 224),
        patch_size: Tuple[int, int, int] = (2, 14, 14),
        estimate_vis: bool = False,
        estimate_depth: bool = False,
        sam_head_depth: int = 2,
        decoding_out_dim_factor: int = 8,
        num_prompt_points: int = 2,
        num_point_embeddings: int = 2,
        modify_pointlabels_for_windowing: bool = False,
        prompt_using_features: bool = False,
        attend_to_past: bool = False,
        depth_fn: str = "linear",
        vis_fn: str = "linear",
        estimation_directions: List[Literal[1, -1]] = [1, -1],
        max_queries: int = 192,
    ):
        """SAM-based tracking head for sliding window 2D/3D tracking for video input

        Args:
            task_name (str, optional): task name. Defaults to "track_2d".
            prompt_embed_dim (int, optional): Embedding dimension of prompt, same as video encoder embedding dimension. Defaults to 1408.
            image_size (Tuple[int, int, int], optional): input video size. Defaults to (16, 224, 224).
            patch_size (Tuple[int, int, int], optional): video patch size used for video tokenization. Defaults to (2, 14, 14).
            estimate_vis (bool, optional): Estimate visibility. Defaults to False.
            estimate_depth (bool, optional): Estimate depth. Defaults to False.
            sam_head_depth (int, optional): Number of two-way attention layers. Defaults to 2.
            decoding_out_dim_factor (int, optional): the factor to decode the mask out dimension. Defaults to 8.
            num_prompt_points (int, optional): offset used to get the token id for prompt features. Defaults to 2.
            num_point_embeddings (int, optional): Number of point embeddings for different states of the input points. Defaults to 2.
            modify_pointlabels_for_windowing (bool, optional): Modify point labels for windowing. Defaults to False.
            prompt_using_features (bool, optional): Prompt tracking using track features. Defaults to False.
            attend_to_past (bool, optional): Attend to past video tokens for sliding window tracking. Defaults to False.
            depth_fn (str, optional): Depth function for depth estimation. Defaults to "linear".
            vis_fn (str, optional): Visibility function for visibility estimation. Defaults to "linear".
            estimation_directions (List[Literal[1,, optional): Which direction to perform tracking. Defaults to [1, -1].
            max_queries (int, optional): Large number of queries are handled by linear processing of smaller batches of size max_queries. Defaults to 192.
        """

        super(VideoMAETrack2DSamHead, self).__init__()

        self.task_name = task_name
        self.prompt_embed_dim = prompt_embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.estimate_vis = estimate_vis
        self.estimate_depth = estimate_depth
        self.sam_head_depth = sam_head_depth
        self.decoding_out_dim_factor = decoding_out_dim_factor
        self.num_prompt_points = num_prompt_points
        self.num_point_embeddings = num_point_embeddings
        self.modify_pointlabels_for_windowing = modify_pointlabels_for_windowing
        self.prompt_using_features = prompt_using_features
        self.attend_to_past = attend_to_past
        self.depth_fn = depth_fn
        self.vis_fn = vis_fn
        self.estimation_directions = estimation_directions
        self.max_queries = max_queries

        # comput number of mask token and their id
        self.num_mask_tokens = 0
        self.token_ids = {}  # specifies id for io tokens from two-way transformer
        self.token_ids["xy"] = self.num_mask_tokens
        self.num_mask_tokens += 1
        if self.estimate_vis:
            self.token_ids["vis"] = self.num_mask_tokens
            self.num_mask_tokens += 1
        if self.estimate_depth:
            self.token_ids["depth"] = self.num_mask_tokens
            self.num_mask_tokens += 1

        self.image_embedding_size = (
            int(image_size[0] / patch_size[0]),
            int(image_size[1] / patch_size[1]),
            int(image_size[2] / patch_size[2]),
        )
        self.video_tokens_size = (
            self.image_embedding_size[0] * self.image_embedding_size[1] * self.image_embedding_size[2]
        )

        # define the point embedding
        self.prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=self.image_embedding_size,
            input_image_size=image_size,
            prompt_using_features=self.prompt_using_features,
            num_point_embeddings=self.num_point_embeddings,
        )
        self.mask_decoder = MaskDecoder(
            transformer=TwoWayTransformer(
                depth=self.sam_head_depth,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            num_mask_tokens=self.num_mask_tokens,
            decoding_out_dim_factor=decoding_out_dim_factor,
        )

        # save the grid xy for softargmax for the 2D track position estimation
        self.register_buffer("grid_xy", self.get_meshgrid(), persistent=False)

        # add additional linear layer before passing to next window
        if self.prompt_using_features:
            # output mask tokens are followed by sparse point prompt which is then followed by track feature prompt
            self.token_ids["prompt_feat"] = self.num_mask_tokens + self.num_prompt_points
            self.prompt_feature_linear_layer = torch.nn.Linear(self.prompt_embed_dim, self.prompt_embed_dim)

        # define mask token for attending to past
        if self.attend_to_past:
            # mask token used for non-overlapping part for sliding window memory mechanism
            self.processed_video_mask_token = torch.nn.Embedding(1, self.prompt_embed_dim)
            self.processed_video_features_proj = torch.nn.Linear(self.prompt_embed_dim, self.prompt_embed_dim)

        self.task_suffix = "_track_2d"

    def get_meshgrid(self) -> torch.Tensor:
        """Gets meshgrid xy using image size"""
        x = torch.arange(self.image_size[2]).to(dtype=torch.float32)
        y = torch.arange(self.image_size[1]).to(dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
        grid_xy_2hw = torch.cat([grid_x.unsqueeze(0), grid_y.unsqueeze(0)], dim=0)
        grid_xy_2hw += 0.5  # offset to pixel center
        return grid_xy_2hw

    def softargmax(self, logits: torch.Tensor):
        B, N, T, H, W = logits.shape
        heatmap = torch.nn.functional.softmax(logits.view(B, N, T, 1, H * W), dim=-1)
        grid_xy = self.grid_xy.view(2, -1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        xy_est = torch.sum(heatmap * grid_xy, dim=-1, keepdim=False)
        heatmap_xy = heatmap.view(B, N, T, H, W)
        return (xy_est, heatmap_xy)

    def estimate_visibility(self, logits: torch.Tensor) -> torch.Tensor:
        visibility_bnt = torch.mean(logits, dim=[-1, -2], keepdim=False)
        visibility_bnt = apply_fn(visibility_bnt, self.vis_fn)
        return visibility_bnt

    def forward_windowed(
        self,
        enc_features_bpc_2dlist: List[List[torch.Tensor]],
        track_2d_pointquerries_bn3: torch.Tensor,
        track_2d_pointlabels_bn: torch.Tensor,
        time_strides: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Handling large number of point qurries by using a for loop.

        Args:
            enc_features_bpc_2dlist (List[List[torch.Tensor]]): Encoder features list for each window. Features are of shape (batch, num_tokens, embed_dim)
            track_2d_pointquerries_bn3 (torch.Tensor): Point queries of shape (batch, num_points, 3)
            track_2d_pointlabels_bn (torch.Tensor): Point labels of shape (batch, num_points)
            time_strides (Optional[torch.Tensor], optional): Time strides for each window. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: Output for the tracking head
        """
        if track_2d_pointquerries_bn3.shape[1] < self.max_queries:
            return self.forward_windowed_core(
                enc_features_bpc_2dlist=enc_features_bpc_2dlist,
                track_2d_pointquerries_bn3=track_2d_pointquerries_bn3,
                track_2d_pointlabels_bn=track_2d_pointlabels_bn,
                time_strides=time_strides,
                **kwargs,
            )
        else:
            N = track_2d_pointquerries_bn3.shape[1]

            out_list = []
            total_loops = int(math.ceil(N / self.max_queries))
            for i in range(total_loops):
                out = self.forward_windowed_core(
                    enc_features_bpc_2dlist=enc_features_bpc_2dlist,
                    track_2d_pointquerries_bn3=track_2d_pointquerries_bn3[
                        :, i * self.max_queries : (i + 1) * self.max_queries
                    ],
                    track_2d_pointlabels_bn=track_2d_pointlabels_bn[
                        :, i * self.max_queries : (i + 1) * self.max_queries
                    ],
                    time_strides=time_strides,
                    **kwargs,
                )
                out_list.append(out)

            out = {}
            for key in out_list[0].keys():
                out[key] = torch.cat([out_curr[key] for out_curr in out_list], dim=1)
            return out

    def forward_windowed_core(
        self,
        enc_features_bpc_2dlist: List[List[torch.Tensor]],
        track_2d_pointquerries_bn3: torch.Tensor,
        track_2d_pointlabels_bn: torch.Tensor,
        time_strides: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Core function for forward pass for windowed tracking, check forward_windowed for more details"""

        # if windowing not needed just do a forward pass
        if time_strides is None:
            return self.forward(
                enc_features_bpc_2dlist[0], track_2d_pointquerries_bn3, track_2d_pointlabels_bn
            )

        # buffers to get combined results for all windows
        dtype = track_2d_pointquerries_bn3.dtype
        device = track_2d_pointquerries_bn3.device
        window_size = self.image_size[0]
        B = track_2d_pointquerries_bn3.shape[0]
        N = track_2d_pointquerries_bn3.shape[1]
        T = int(time_strides[-1] + window_size)
        traj_est_bn2t = torch.zeros(B, N, 2, T, dtype=dtype, device=device)  # 2D track
        vis_est_bn1t = -torch.ones(B, N, 1, T, dtype=dtype, device=device) * 10.0  # visibility
        if self.estimate_depth:
            depth_est_bn1t = torch.zeros(B, N, 1, T, dtype=dtype, device=device)  # depth

        assert B == 1, "Currently only supports batch size of 1"
        assert len(self.estimation_directions) == 1 and self.estimation_directions[0] == 1, (
            "Currently only positive direction estimation is supported for sliding window tracking."
            "Run twice, with and without video flipping, and then combine outputs."
        )

        # sign 1 indicates estimate future traj and sign -1 indidicates estimate past traj
        for sign in self.estimation_directions:

            # #########################################################
            # initialize everything for causal estimation
            # #########################################################

            # initialize prompt features and prompt feature labels to zero
            if self.prompt_using_features:
                prompt_features_bnc = torch.zeros(B, N, self.prompt_embed_dim, dtype=dtype, device=device)
                prompt_feature_lables_bn = torch.zeros(B, N, dtype=dtype, device=device)
            else:
                prompt_features_bnc, prompt_feature_lables_bn = None, None

            # initialize history features using the learned mask token
            if self.attend_to_past:
                enc_history_features_bnpc = (
                    self.processed_video_mask_token.weight[0][None, None, None, :]
                ).repeat(B, N, self.video_tokens_size, 1)
            else:
                enc_history_features_bnpc = torch.zeros(
                    B, N, self.video_tokens_size, self.prompt_embed_dim, dtype=dtype, device=device
                )
            curr_track_2d_pointquerries_bn3 = track_2d_pointquerries_bn3.clone()
            curr_track_2d_pointlabels_bn = track_2d_pointlabels_bn.clone()

            ########################################################################################
            # perform sliding window tracking
            ########################################################################################

            for win_id in range(time_strides.shape[0]):

                ####################################################################################
                # prepare the time offsets needed for the current window
                ####################################################################################

                # get the win_id based on the direction of sliding
                win_id = win_id if sign == 1 else time_strides.shape[0] - 1 - win_id

                # get start times for current and next window
                if sign > 0:
                    curr_start_time_id = time_strides[win_id]
                    next_start_time_id = (
                        time_strides[win_id + 1]
                        if win_id < time_strides.shape[0] - 1
                        else time_strides[win_id - 1]
                    )
                else:
                    curr_start_time_id = time_strides[win_id]
                    next_start_time_id = time_strides[win_id - 1] if win_id >= 1 else curr_start_time_id

                ####################################################################################
                # prepare point querries and lables for the current window
                ####################################################################################

                # shift time to current window and get valid results location
                curr_pointquerries_time_offset_bn3 = curr_track_2d_pointquerries_bn3.clone()

                # Valid results time instances happen after query time instance
                valid_results_bn1t = (
                    torch.arange(window_size).repeat(B, N, 1).to(device=device) + curr_start_time_id + 0.5
                )
                if sign == 1:
                    valid_results_bn1t = (
                        valid_results_bn1t - curr_pointquerries_time_offset_bn3[:, :, 0:1]
                    ) >= 0
                else:
                    valid_results_bn1t = (
                        valid_results_bn1t - curr_pointquerries_time_offset_bn3[:, :, 0:1]
                    ) <= 0
                valid_results_bn1t = valid_results_bn1t[:, :, None, :]
                # Are the result valid in this window
                valid_results_bn = torch.sum(valid_results_bn1t, dim=-1)[..., 0] > 0

                # set correct time reference
                curr_pointquerries_time_offset_bn3[:, :, 0] -= curr_start_time_id

                # Figure out if the point querries are invalid, input or predicted
                # label 0 for point querries that are meaningless (they are passed for batch processing but not valid)
                curr_track_2d_pointlabels_bn[~valid_results_bn] = 0
                curr_track_2d_pointlabels_bn[valid_results_bn] = 1
                if self.modify_pointlabels_for_windowing:
                    # label 1 if the point qurries are provide as input
                    curr_querries_equals_input = curr_track_2d_pointquerries_bn3 == track_2d_pointquerries_bn3
                    valid_input_querries = torch.sum(curr_querries_equals_input, dim=-1) > 0
                    curr_track_2d_pointlabels_bn[valid_input_querries] = 1
                    # label 2 if the point qurries are estimated from previous windows
                    valid_estimated_queries = torch.logical_and(valid_results_bn, ~valid_input_querries)
                    curr_track_2d_pointlabels_bn[valid_estimated_queries] = 2

                ####################################################################################
                # Prepare input to enable memory mechanism
                ####################################################################################
                # Add decoded features from previous windows to enable memory mechanism
                if self.attend_to_past:
                    curr_enc_features = (
                        enc_features_bpc_2dlist[win_id][-1].unsqueeze(1) + enc_history_features_bnpc
                    )
                else:
                    curr_enc_features = enc_features_bpc_2dlist[win_id][-1]

                ####################################################################################
                # Do a forward pass for current window
                ####################################################################################
                assert not curr_pointquerries_time_offset_bn3.requires_grad, "point_querries no grad"
                assert not curr_track_2d_pointlabels_bn.requires_grad, "point_labels no grad"
                if prompt_feature_lables_bn is not None:
                    assert not prompt_feature_lables_bn.requires_grad, "point_labels no grad"
                curr_out = self.forward(
                    enc_features_bpc_list=[curr_enc_features],
                    track_2d_pointquerries_bn3=curr_pointquerries_time_offset_bn3,
                    track_2d_pointlabels_bn=curr_track_2d_pointlabels_bn,
                    track_2d_promptfeatures_bnc=prompt_features_bnc,
                    track_2d_promptfeaturelabels_bn=prompt_feature_lables_bn,
                )

                ####################################################################################
                # Write the current window results to a buffer
                ####################################################################################
                vis_est_bn1t[..., curr_start_time_id : curr_start_time_id + window_size][
                    valid_results_bn1t
                ] = curr_out[f"{self.task_name}_vis_est_bn1t"][valid_results_bn1t].to(dtype)
                traj_est_bn2t[..., 0:1, curr_start_time_id : curr_start_time_id + window_size][
                    valid_results_bn1t
                ] = curr_out[f"{self.task_name}_traj_est_bn2t"][:, :, 0:1][valid_results_bn1t].to(dtype)
                traj_est_bn2t[..., 1:2, curr_start_time_id : curr_start_time_id + window_size][
                    valid_results_bn1t
                ] = curr_out[f"{self.task_name}_traj_est_bn2t"][:, :, 1:2][valid_results_bn1t].to(dtype)

                if self.estimate_depth:
                    depth_est_bn1t[..., curr_start_time_id : curr_start_time_id + window_size][
                        valid_results_bn1t
                    ] = curr_out[f"{self.task_name}_depth_est_bn1t"][valid_results_bn1t].to(dtype)

                if (sign > 0 and win_id == time_strides.shape[0] - 1) or (sign < 0 and win_id == 0):
                    continue

                ####################################################################################
                # Prepare for next window
                ####################################################################################

                # Update the prompt features and prompt feature labels to be used for the next window
                if self.prompt_using_features:
                    prompt_features_bnc[valid_results_bn] = curr_out[f"{self.task_name}_prompt_features_bnc"].to(dtype)[  # type: ignore
                        valid_results_bn
                    ]
                    prompt_feature_lables_bn[valid_results_bn] = 1  # type: ignore

                # Figure out the overlap start and stop time for preparing decoded video tokens for next window
                if sign == 1:
                    overlap_start_time_id = next_start_time_id
                    overlap_stop_time_id = curr_start_time_id + window_size
                    offset = next_start_time_id
                else:
                    overlap_start_time_id = curr_start_time_id
                    overlap_stop_time_id = next_start_time_id + window_size
                    offset = curr_start_time_id

                # Prepare decoded video tokens to be used for the next window using the memory mechanism
                if self.attend_to_past:
                    enc_history_features_bnpc = curr_out["track_2d_enc_features_with_track_history_bnpc"].to(
                        dtype
                    )
                    enc_history_features_bncthw = einops.rearrange(
                        enc_history_features_bnpc,
                        "b n (t h w) c -> b n c t h w",
                        t=self.image_embedding_size[0],
                        h=self.image_embedding_size[1],
                        w=self.image_embedding_size[2],
                    )
                    # mask non-overlapping part
                    enc_history_mask_features_bncthw = self.processed_video_mask_token.weight[0][
                        None, None, :, None, None, None
                    ].repeat(
                        B,
                        N,
                        1,
                        int(self.image_embedding_size[0] / 2),
                        self.image_embedding_size[1],
                        self.image_embedding_size[2],
                    )
                    if sign == 1:
                        enc_history_valid_features_bncthw = enc_history_features_bncthw[
                            :, :, :, int(self.image_embedding_size[0] / 2) :, :, :
                        ]
                        enc_history_features_bncthw = torch.cat(
                            [enc_history_valid_features_bncthw, enc_history_mask_features_bncthw], dim=3
                        )
                    else:
                        enc_history_valid_features_bncthw = enc_history_features_bncthw[
                            :, :, :, : int(self.image_embedding_size[0] / 2), :, :
                        ]
                        enc_history_features_bncthw = torch.cat(
                            [enc_history_mask_features_bncthw, enc_history_valid_features_bncthw], dim=3
                        )
                    enc_history_features_bnpc = einops.rearrange(
                        enc_history_features_bncthw,
                        "b n c t h w -> b n (t h w) c",
                        t=self.image_embedding_size[0],
                        h=self.image_embedding_size[1],
                        w=self.image_embedding_size[2],
                    )

                ####################################################################################
                # Get new point qurries for the next window based on the current estimations
                ####################################################################################

                # detach the gradients for the current estimations
                curr_vis_est_overlap_bn1t = (
                    vis_est_bn1t[..., overlap_start_time_id:overlap_stop_time_id].clone().detach()
                )
                curr_traj_est_overlap_bn2t = (
                    traj_est_bn2t[..., overlap_start_time_id:overlap_stop_time_id].clone().detach()
                )
                _, N_curr, _, _ = curr_vis_est_overlap_bn1t.shape

                # Sample a point on the track in the region that overlaps with the next window
                best_vis_id_bn1 = torch.argmax(curr_vis_est_overlap_bn1t, dim=-1)
                new_querries_bn3 = []
                for i in range(N_curr):
                    curr_xy = curr_traj_est_overlap_bn2t[0, i, :, best_vis_id_bn1[0, i, 0]]
                    new_querries_bn3.append(
                        torch.Tensor(
                            [
                                best_vis_id_bn1[0, i, 0] + offset + 0.5,
                                curr_xy[0],
                                curr_xy[1],
                            ]
                        )
                        .to(dtype=dtype)
                        .to(device=device)[None, None]
                    )
                new_querries_bn3 = torch.cat(new_querries_bn3, dim=1)

                # Use the new qurries only if they are after/before the provided querries
                if sign == 1:  # after
                    valid_querries = new_querries_bn3[0, :, 0] > curr_track_2d_pointquerries_bn3[0, :, 0]
                else:  # before
                    valid_querries = new_querries_bn3[0, :, 0] < curr_track_2d_pointquerries_bn3[0, :, 0]
                curr_track_2d_pointquerries_bn3[0, valid_querries, :] = new_querries_bn3[0, valid_querries, :]

        out = {
            f"{self.task_name}_traj_est_bn2t": traj_est_bn2t,
            f"{self.task_name}_vis_est_bn1t": vis_est_bn1t,
        }
        if self.estimate_depth:
            out[f"{self.task_name}_depth_est_bn1t"] = depth_est_bn1t

        return out

    def forward(
        self,
        enc_features_bpc_list: List[torch.Tensor],  # TODO: Fix name to also allow bnpc
        track_2d_pointquerries_bn3: torch.Tensor,
        track_2d_pointlabels_bn: torch.Tensor,
        track_2d_promptfeatures_bnc: Optional[torch.Tensor] = None,
        track_2d_promptfeaturelabels_bn: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass for the tracking head for a single window.
        Tracking is prompted using the point queries and their labels, as well as optional prompt features and prompt feature labels.

        Args:
            enc_features_bpc_list (List[torch.Tensor]): Encoded features for each window, feature shape is (batch, num_tokens, embed_dim)
            track_2d_pointquerries_bn3 (torch.Tensor): Point queries for each window, shape is (batch, num_points, 3)
            track_2d_pointlabels_bn (torch.Tensor): Point labels for each window, shape is (batch, num_points)
            track_2d_promptfeatures_bnc (Optional[torch.Tensor], optional): Optional prompt features for each window, shape is (batch, num_points, embed_dim)
            track_2d_promptfeaturelabels_bn (Optional[torch.Tensor], optional): Optional prompt feature labels for each window, shape is (batch, num_points)

        Returns:
            Dict[str, torch.Tensor]: Output for the tracking head
        """

        enc_features = enc_features_bpc_list[-1]
        # Repeat video token features for each query point bpc -> bnpc
        if enc_features.dim() == 3:
            enc_features = enc_features.unsqueeze(1)
        point_coords = track_2d_pointquerries_bn3.unsqueeze(-2)
        point_labels = track_2d_pointlabels_bn.unsqueeze(-1)

        prompt_features = (
            track_2d_promptfeatures_bnc.unsqueeze(-2) if track_2d_promptfeatures_bnc is not None else None
        )
        prompt_feature_labels = (
            track_2d_promptfeaturelabels_bn.unsqueeze(-1)
            if track_2d_promptfeaturelabels_bn is not None
            else None
        )

        batch_size = enc_features.shape[0]
        logits = []
        prompt_features_updated = []
        enc_features_updated = []

        for i in range(batch_size):
            logits_i, prompt_feature_updated_i, enc_features_updated_i = self.forward_single_batch(
                enc_features=enc_features[i : i + 1],
                point_coords=point_coords[i],
                point_labels=point_labels[i],
                prompt_feature=prompt_features[i] if prompt_features is not None else None,
                prompt_feature_label=prompt_feature_labels[i] if prompt_feature_labels is not None else None,
            )
            if logits_i is not None:
                logits.append(logits_i[None])
            if prompt_feature_updated_i is not None:
                prompt_features_updated.append(prompt_feature_updated_i[None])
            if enc_features_updated_i is not None:
                enc_features_updated.append(enc_features_updated_i[None])

        logits = torch.cat(logits, dim=0)
        prompt_features_updated = (
            torch.cat(prompt_features_updated, dim=0) if len(prompt_features_updated) > 0 else None
        )
        enc_features_updated = (
            torch.cat(enc_features_updated, dim=0) if len(enc_features_updated) > 0 else None
        )

        # save features for next window
        out = {}
        if prompt_features_updated is not None:
            out.update({f"{self.task_name}_prompt_features_bnc": prompt_features_updated[:, :, 0]})
        if enc_features_updated is not None:
            out.update({f"{self.task_name}_enc_features_with_track_history_bnpc": enc_features_updated})

        # POSTPROCESSING FOR FINAL OUTPUT
        # compute softargmax
        xy, _ = self.softargmax(logits[:, :, self.token_ids["xy"], :, :, :])
        xy_bn2t = xy.permute(0, 1, 3, 2)
        out.update({f"{self.task_name}_traj_est_bn2t": xy_bn2t})

        if self.estimate_vis:
            visibility_bn1t = self.estimate_visibility(
                logits[:, :, self.token_ids["vis"], :, :, :]
            ).unsqueeze(2)
            out[f"{self.task_name}_vis_est_bn1t"] = visibility_bn1t

        if self.estimate_depth:
            depth_bnt = torch.mean(
                logits[:, :, self.token_ids["depth"], :, :, :], dim=[-1, -2], keepdim=False
            )
            depth_bn1t = apply_fn(depth_bnt, self.depth_fn).unsqueeze(2)
            out[f"{self.task_name}_depth_est_bn1t"] = depth_bn1t

        return out

    def forward_single_batch(
        self,
        enc_features: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        prompt_feature: Optional[torch.Tensor] = None,
        prompt_feature_label: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for the tracking head for a single batch.

        Args:
            enc_features (torch.Tensor): Encoded features, shape is (1, num_points, num_tokens, embed_dim)
            point_coords (torch.Tensor): Point coordinates, shape is (num_points, 1, 3)
            point_labels (torch.Tensor): Point labels, shape is (num_points, 1)
            prompt_feature (Optional[torch.Tensor], optional): Optional prompt features, shape is (num_points, 1, embed_dim)
            prompt_feature_label (Optional[torch.Tensor], optional): Optional prompt feature labels, shape is (num_points, 1)

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: Output for the tracking head
            out: Predicted masks, shape is (num_points, num_mask_tokens, 16, 224, 224)
            processed_prompt_features: Processed prompt features, shape is (num_points, 1, embed_dim)
            processed_enc_features: Processed video tokens for each prompt point, shape is (num_points, num_tokens, embed_dim)
        """

        # Prepare point and track feature prompts
        points = (point_coords, point_labels)
        features = None
        dtype = point_coords.dtype
        device = point_coords.device
        if self.prompt_using_features:
            if prompt_feature is None:
                prompt_feature = (
                    torch.zeros(point_coords.shape[0], 1, self.prompt_embed_dim)
                    .to(dtype=dtype)
                    .to(device=device)
                )
            if prompt_feature_label is None:
                prompt_feature_label = torch.zeros(point_coords.shape[0]).to(dtype=dtype).to(device=device)
            features = (prompt_feature, prompt_feature_label)

        # Embed prompts
        sparse_embeddings = self.prompt_encoder(points=points, features=features)  # [1, 2, 1408]

        # Predict masks using the mask decoder
        out, processed_features = self.mask_decoder(
            image_embeddings=enc_features,  # [1, 2048, 1408] or [N, 2048, 1408]
            image_pe=self.prompt_encoder.get_dense_pe(),  # [1, 1408, 8, 16, 16]
            sparse_prompt_embeddings=sparse_embeddings,
        )

        # Upsample mask
        out = torch.nn.functional.interpolate(
            out, size=self.image_size, mode="trilinear", align_corners=False
        )  # [1, num_mask_tokens, 16, 224, 224]

        # Process prompt tokens
        processed_prompt_features = (
            self.prompt_feature_linear_layer(
                processed_features["io_features"][
                    :, self.token_ids["prompt_feat"] : self.token_ids["prompt_feat"] + 1, :
                ]
            )
            if self.prompt_using_features
            else None
        )

        # Process enc features
        processed_enc_features = (
            self.processed_video_features_proj(processed_features["enc_features"])
            if self.attend_to_past
            else None
        )

        return out, processed_prompt_features, processed_enc_features
