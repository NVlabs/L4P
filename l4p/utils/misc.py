# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

from typing import Tuple
import torch


def apply_fn(x: torch.Tensor, fn_type: str = "linear") -> torch.Tensor:
    """Applies a function to a tensor.

    Args:
        x (torch.Tensor): The tensor to apply the function to.
        fn_type (str): The type of function to apply.

    Returns:
        torch.Tensor: The tensor with the function applied.
    """
    if fn_type == "log":
        out = torch.log(x)
    elif fn_type == "exp":
        out = torch.exp(x)
    elif fn_type == "sigmoid":
        out = torch.nn.functional.sigmoid(x)
    elif fn_type == "linear":
        out = x
    elif fn_type == "inverse":
        eps = 1e-8
        mask = x.abs() > eps
        out = torch.zeros_like(x)
        out[mask] = (1.0 / x[mask]).to(x.dtype)
    else:
        print(f"Not implemented {fn_type}")
        raise NotImplementedError

    return out.to(x.dtype)


def check_inf_nan(x: torch.Tensor) -> Tuple[bool, torch.Tensor]:
    """Returns if inf or nan found and a bool tensor indicating locations"""
    nan_inf_map = torch.logical_or(torch.isnan(x), torch.isinf(x))
    found_inf_nan = True if torch.sum(nan_inf_map) > 0 else False
    return found_inf_nan, nan_inf_map


def safe_inverse(depth_or_disp, keep_above=0.0):
    """Returns the inverse of a tensor, with a minimum value of keep_above.

    Args:
        depth_or_disp (torch.Tensor): The tensor to apply the function to.
        keep_above (float): The minimum value to keep.

    Returns:
        torch.Tensor: The tensor with the function applied.
    """
    assert isinstance(depth_or_disp, torch.Tensor)
    disp_or_depth = torch.zeros_like(depth_or_disp)
    non_negtive_mask = depth_or_disp > keep_above
    disp_or_depth[non_negtive_mask] = (1.0 / depth_or_disp[non_negtive_mask]).to(disp_or_depth.dtype)
    return disp_or_depth
