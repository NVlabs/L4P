# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import yaml
import torch
from jsonargparse import ArgumentParser, Namespace
from lightning.pytorch import LightningModule
from lightning.fabric import Fabric
from typing import Optional


def prepare_model(
    model_config_path: str,
    ckpt_path: str,
    max_queries: Optional[int] = None,
    precision: str = "16-mixed",
    accelerator: str = "gpu",
):
    """
    Loads pytorch lightning model using model config file and checkpoint file.
    Optionally, sets maximum number of queries for track_2d head.

    Args:
        model_config_path (str): Path to model configuration file.
        ckpt_path (str): Path to checkpoint file.
        max_queries (Optional[int], optional): Maximum number of queries. Defaults to None.
        precision (str, optional): Precision. Defaults to "16-mixed".
        accelerator (str, optional): Accelerator. Defaults to "gpu".

    Returns:
        _type_: Returns pytorch lightning model.
    """
    # get model paramters
    with open(model_config_path, "r") as f:
        model_dict = yaml.safe_load(f)
    model_ns = Namespace({"model": model_dict})

    if max_queries is not None:
        model_ns["model"]["init_args"]["l4p_model"]["init_args"]["task_heads"]["init_args"]["modules"][
            "track_2d"
        ]["init_args"]["max_queries"] = max_queries

    # instantiate the model class
    parser = ArgumentParser()
    parser.add_argument("--model", type=LightningModule)
    model = parser.instantiate_classes(model_ns).model

    # load state dict
    state_dict = torch.load(ckpt_path, weights_only=True)["state_dict"]
    model.load_state_dict(state_dict)
    model = model.eval()

    # convert to appropriate type
    fabric = Fabric(precision=precision, accelerator=accelerator)  # type: ignore
    model = fabric.setup(model)

    return model
