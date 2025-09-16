# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

from lightning.pytorch.cli import LightningCLI


def cli_main():
    cli = LightningCLI(save_config_kwargs={"overwrite": True})  # noqa: F841


if __name__ == "__main__":
    cli_main()
