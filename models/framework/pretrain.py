import time
from typing import Optional

import torch
from utils.metrics import lat_weighted_mse
from optim.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning import LightningModule

 
class ForecastPretrain(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        restart_path: Optional[str],
        lr: float,
        beta_1: float,
        beta_2: float,
        weight_decay: float,
        warmup_steps: int = 5,
        max_steps: int = 100,
        warmup_start_lr: float = 0.0001,
        eta_min: float = 1e-5,
        opt_name: str = "adamw",
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        if restart_path is not None and len(restart_path) > 0:
            self.load_from_checkpoint(restart_path)
        self._last_time: Optional[float] = None

    def forward(self, single_inputs, atmos_inputs, lead_times, metadata):
        _, single_out, atmos_out = self.net(
            single_inputs,
            atmos_inputs,
            lead_times,
            metadata,
        )
        return single_out, atmos_out

    def training_step(self, batch, batch_idx: int):
        single_inputs, atmos_inputs, single_targets, atmos_targets, lead_times, metadata = batch
        loss_dict, _, _ = self.net(
            single_inputs,
            atmos_inputs,
            lead_times,
            metadata,
            single_targets,
            atmos_targets,
            metric=[lat_weighted_mse],
        )
        # Log loss metrics.
        # TODO(Cris): Eventually move logs into a lightning callback or a more appropiate place.
        batch_size = single_inputs.shape[0]
        for k, v in loss_dict[0].items():
            self.log(
                f"train/{k}", v, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=False
            )
        self._log_iteration_performance(single_inputs, atmos_inputs, single_targets, atmos_targets)

        return loss_dict[0]["loss"]

    def _log_iteration_performance(
        self, single_inputs, atmos_inputs, single_targets, atmos_targets
    ):
        # Log training loop and data loading performance.
        if self._last_time is not None:
            now = time.time()
            dur = now - self._last_time
            self.log("train/seconds_per_iteration", dur, on_step=True, prog_bar=False)
            num_floats = single_inputs.numel() + sum([v.numel() for v in atmos_inputs.values()], 0)
            num_floats += single_targets.numel() + sum(
                [v.numel() for v in atmos_targets.values()], 0
            )
            self.log(
                "train/data_loading_throughput",
                32 * num_floats / dur / 1e9,
                on_step=True,
                prog_bar=False,
            )
            self._last_time = now
        else:
            self._last_time = time.time()

    def validation_step(self, batch, batch_idx: int):
        single_inputs, atmos_inputs, single_targets, atmos_targets, lead_times, metadata = batch
        loss_dict, results = self.net.evaluate(
            single_inputs,
            atmos_inputs,
            lead_times,
            metadata,
            single_targets,
            atmos_targets,
            metrics=[lat_weighted_mse],
        )
        # Log loss metrics.
        # During validation, compared to speed, we focus more on accurate metric. So we set sync_dist=true.
        batch_size = single_inputs.shape[0]
        for k, v in loss_dict[0].items():
            self.log(
                f"val/{k}",
                v,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return results

    def test_step(self, batch, batch_idx: int):
        single_inputs, atmos_inputs, single_targets, atmos_targets, lead_times, metadata = batch
        loss_dict, results = self.net.evaluate(
            single_inputs,
            atmos_inputs,
            lead_times,
            metadata,
            single_targets,
            atmos_targets,
            metrics=[lat_weighted_mse],
        )
        # Log loss metrics.
        # During testing, compared to speed, we focus more on accurate metric. So we set sync_dist=true.
        batch_size = single_inputs.shape[0]
        for k, v in loss_dict[0].items():
            self.log(
                f"test/{k}",
                v,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return results

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if "pos_embed" in name:
                no_decay.append(param)
            # elif "bias" in name:
            #     no_decay.append(param)
            else:
                decay.append(param)

        if self.hparams.opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": self.hparams.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=self.hparams.lr,
                betas=(self.hparams.beta_1, self.hparams.beta_2),
            )
        else:
            raise NotImplementedError(f"Optimizer {self.hparams.opt_name} not implemented")

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_steps,
            max_epochs=self.hparams.max_steps,
            warmup_start_lr=self.hparams.warmup_start_lr,
            eta_min=self.hparams.eta_min,
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
