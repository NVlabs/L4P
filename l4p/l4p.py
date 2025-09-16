# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

from typing import Optional, Dict, Any, List
import torch
import lightning as L


class L4PLitModule(L.LightningModule):
    def __init__(
        self,
        tasks: List["str"],
        l4p_model: torch.nn.Module,
        loss_module: Optional[torch.nn.Module] = None,
        metrics_module: Optional[torch.nn.Module] = None,
        optimizer_opts: Optional[Dict[str, Any]] = None,
        scheduler_opts: Optional[Dict[str, Any]] = None,
        strict_loading: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tasks = tasks
        self.l4p_model = l4p_model
        self.loss_module = loss_module
        self.metrics_module = metrics_module
        self.optimizer_opts = optimizer_opts
        self.scheduler_opts = scheduler_opts
        # lightning parameters
        self.strict_loading = strict_loading

        return

    def forward(self, batch, tasks):

        return self.l4p_model.forward(batch, tasks)

    def do_data_sanity_checks(self, batch, phase):

        skip = False
        if not phase == "train":
            return skip

        if "track_2d_valid_bn1t" in batch.keys():
            if torch.sum(batch["track_2d_valid_bn1t"]) == 0:
                skip = True
                print("skipping due to invalid track")

        return skip

    def step(self, phase, batch, batch_idx):

        for key, _ in batch.items():
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device=self.device)

        skip = False

        # forward pass
        out = self.forward(batch, self.tasks)

        if phase == "predict":
            return out

        # compute loss
        loss, loss_dict, metadata = (
            self.loss_module(batch, out) if self.loss_module is not None else (0, {}, {})
        )

        # logging
        with torch.inference_mode():
            # compute additional metrics
            metrics_dict = {}
            if self.metrics_module is not None:
                metrics_dict, _ = self.metrics_module(batch, out, metadata)

            # log everything
            log = {}
            log[f"scalars/{phase}/loss"] = (
                torch.clone(loss).to(torch.float32) if torch.is_tensor(loss) else loss
            )

            for loss_key in loss_dict.keys():
                log[f"scalars/{phase}/{loss_key}"] = torch.clone(loss_dict[loss_key]).to(torch.float32)
            for metric_key in metrics_dict.keys():
                log[f"scalars/{phase}/{metric_key}"] = torch.clone(metrics_dict[metric_key]).to(torch.float32)

            self.log_dict(log)

        return loss, out, skip

    def training_step(self, batch, batch_idx):
        loss, out, skip = self.step("train", batch, batch_idx)
        return {"loss": loss, "out": out} if not skip else None

    def validation_step(self, batch, batch_idx):
        loss, out, skip = self.step("val", batch, batch_idx)
        return {"loss": loss, "out": out} if not skip else None

    def test_step(self, batch, batch_idx):
        loss, out, skip = self.step("val", batch, batch_idx)
        return {"loss": loss, "out": out} if not skip else None

    def predict_step(self, batch, batch_idx):
        out = self.step("predict", batch, batch_idx)
        return out

    def configure_optimizers(self):
        """Configuring optmizers and learning rate schedules"""

        out = {}
        out["optimizer"] = None
        out["lr_scheduler"] = None
        if self.optimizer_opts is not None:
            params = [param for param in self.parameters() if param.requires_grad]
            out["optimizer"] = torch.optim.AdamW(params, **(self.optimizer_opts))  # type: ignore
            if self.scheduler_opts is not None:
                lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer=out["optimizer"], **self.scheduler_opts
                )
                out["lr_scheduler"] = {"scheduler": lr_scheduler, "interval": "step"}

        return out
