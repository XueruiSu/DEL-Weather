from typing import Dict

import torch
import numpy as np


def lat_weighted_mse(
    single_pred: torch.Tensor,
    single_targ: torch.Tensor,
    atmos_pred: Dict[str, torch.Tensor],
    atmos_targ: Dict[str, torch.Tensor],
    metadata,
):
    lat = metadata.lat.to(single_pred.device)
    w_lat = torch.cos(torch.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = w_lat.unsqueeze(0).unsqueeze(-1)

    single_error = (single_pred - single_targ) ** 2  # [N, C, H, W]
    atmos_error = {k: (atmos_pred[k] - atmos_targ[k]) ** 2 for k in atmos_pred}  # [N, C, H, W]?

    loss_dict = {}
    w_single_error = single_error * w_lat.unsqueeze(1)
    w_atmos_error = {k: atmos_error[k] * w_lat.unsqueeze(1) for k in atmos_error}
    # TODO figure out the weights
    loss_dict["loss"] = (
        w_single_error.mean() + torch.stack([w_atmos_error[k].mean() for k in w_atmos_error]).mean()
    )
    with torch.no_grad():
        avg_w_single_error = w_single_error.mean(dim=(0, 2, 3))
        for i, var in enumerate(metadata.single_vars):
            loss_dict[var] = avg_w_single_error[i]

        for i, var in enumerate(metadata.atmos_vars):
            avg_w_atmos_error = w_atmos_error[var].mean(dim=(0, 2, 3))
            # TODO confirm order of vars and levels
            for l, level in enumerate(metadata.atmos_vars[var]):
                loss_dict[f"{var}_{level}"] = avg_w_atmos_error[l]

    return loss_dict


def lat_weighted_mse_UQ(
    single_targets: torch.Tensor, atmos_targets: torch.Tensor, 
    single_preds_1: torch.Tensor, atmos_preds_1: torch.Tensor, 
    single_preds_2: torch.Tensor, atmos_preds_2: torch.Tensor, 
    metadata, R2Loss_pre: torch.Tensor,
):  
    '''
    single_targets: torch.Tensor (B, len(single_vars), H, W), 
    atmos_targets: torch.Tensor (B, len(atmos_levels), len(atmos_vars), H, W), 
    single_preds_1: torch.Tensor (B, len(single_vars), H, W), 
    atmos_preds_1: torch.Tensor (B, len(atmos_levels), len(atmos_vars), H, W), 
    single_preds_2: torch.Tensor (B, len(single_vars), H, W), 
    atmos_preds_2: torch.Tensor (B, len(atmos_levels), len(atmos_vars), H, W), 
    metadata: Metadata class,
    R2Loss_pre: torch.Tensor (B, len(single_vars)+len(atmos_levels)*len(atmos_vars)),
    '''
    # preparation:
    lat = metadata.lat.to(single_targets.device)
    w_lat = torch.cos(torch.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = w_lat.unsqueeze(0).unsqueeze(-1)
    num_single_vars = len(metadata.single_vars)
    R2Loss_pre_single_vars = R2Loss_pre[:, :num_single_vars]
    R2Loss_pre_atmos_vars = R2Loss_pre[:, num_single_vars:].reshape(R2Loss_pre.shape[0], -1, len(metadata.atmos_vars))
    
    # residual calculation:
    single_residual = ((single_preds_1 - single_targets) ** 2 - (single_preds_2 - single_targets) ** 2)  # (B, len(single_vars), H, W)
    atmos_residual = (atmos_preds_1 - atmos_targets) ** 2 - (atmos_preds_2 - atmos_targets) ** 2 # (B, len(atmos_levels), len(atmos_vars), H, W)
    w_single_residual = (single_residual * w_lat.unsqueeze(1)).mean(dim=(-2, -1)) # (B, len(single_vars))
    w_atmos_residual = (atmos_residual * w_lat.unsqueeze(1)).mean(dim=(-2, -1)) # (B, len(atmos_levels), len(atmos_vars))
    
    # Loss calculation:
    loss_dict = {}
    w_single_error = (w_single_residual - R2Loss_pre_single_vars) ** 2
    w_atmos_error = (w_atmos_residual - R2Loss_pre_atmos_vars) ** 2
    loss_dict["loss"] = (w_single_error.mean() + w_atmos_error.mean())
    with torch.no_grad():
        avg_w_single_error = w_single_error.mean(dim=(0))
        for i, var in enumerate(metadata.single_vars):
            loss_dict[var] = avg_w_single_error[i]

        for i, var in enumerate(metadata.atmos_vars):
            avg_w_atmos_error = w_atmos_error[:, :, i].mean(dim=(0)) # (len(atmos_levels), )
            # TODO confirm order of vars and levels
            for l, level in enumerate(metadata.atmos_vars[var]):
                loss_dict[f"{var}_{level}"] = avg_w_atmos_error[l]

    return loss_dict


class LatWeightedMSE(torch.nn.Module):
    def __init__(self, lat):
        super().__init__()
        w_lat = np.cos(np.deg2rad(lat))
        w_lat = w_lat / w_lat.mean()  # (H, )
        w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1)
        self.register_buffer("w_lat", w_lat, persistent=False)

    def forward(self, single_pred, single_targ, atmos_pred, atmos_targ, metadata, mask=None):
        self.w_lat = self.w_lat.to(single_pred.device)

        single_error = (single_pred - single_targ) ** 2  # [N, C, H, W]
        atmos_error = (atmos_pred - atmos_targ) ** 2  # [N, L, C, H, W]?

        loss_dict = {}
        if mask is None:
            w_single_error = single_error * self.w_lat.unsqueeze(1)
            w_atmos_error = atmos_error * self.w_lat.unsqueeze(1).unsqueeze(1)
            # TODO figure out the weights
            loss_dict["loss"] = w_single_error.mean() + w_atmos_error.mean()

            with torch.no_grad():
                avg_w_single_error = w_single_error.mean(dim=(0, 2, 3))
                for i, var in enumerate(metadata.single_vars):
                    loss_dict[var] = avg_w_single_error[i]

                avg_w_atmos_error = w_atmos_error.mean(dim=(0, 3, 4))
                # TODO confirm order of vars and levels
                for i, var in enumerate(metadata.atmos_vars):
                    for l, level in enumerate(metadata.atmos_vars[var]):
                        loss_dict[f"{var}_{level}"] = avg_w_atmos_error[i, l]

        else:
            raise NotImplementedError("Masked loss not implemented yet")
            # w_error = (error * self.w_lat.unsqueeze(1) * mask.unsqueeze(1))
            # loss_dict["loss"] = w_error.mean(dim=1).sum() / mask.sum()
            # with torch.no_grad():
            #     avg_w_error = w_error.sum(dim=(0, 2, 3)) / mask.sum()
            #     for i, var in enumerate(vars):
            #         loss_dict[var] = avg_w_error[i]

        return loss_dict


class LatWeightedRMSE(torch.nn.Module):
    def __init__(self, lat, single_transform, atmos_transform):
        super().__init__()
        self.single_transform = single_transform
        self.atmos_transform = atmos_transform
        w_lat = np.cos(np.deg2rad(lat))
        w_lat = w_lat / w_lat.mean()
        w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1)
        self.register_buffer("w_lat", w_lat, persistent=False)

    def forward(self, single_pred, single_targ, atmos_pred, atmos_targ, metadata):
        self.w_lat = self.w_lat.to(single_pred.device)
        single_pred = self.single_transform(single_pred)  # B T C H W
        single_targ = self.single_transform(single_targ)
        atmos_pred = self.atmos_transform(atmos_pred)  # B T C L H W
        atmos_targ = self.atmos_transform(atmos_targ)

        single_error = (single_pred - single_targ) ** 2
        atmos_error = (atmos_pred - atmos_targ) ** 2

        # TODO
        w_single_error = single_error * self.w_lat.unsqueeze(1).unsqueeze(1)
        w_atmos_error = atmos_error * self.w_lat.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        timesteps = atmos_pred.shape[1]
        loss_dict = {}
        for i, var in enumerate(metadata.single_vars):
            for time in timesteps:
                loss_dict[f"{var}_idx_{time}"] = torch.sqrt(w_single_error[:, time, i].mean())

        for i, var in enumerate(metadata.atmos_vars):
            for l, level in enumerate(metadata.atmos_vars[var]):
                for time in timesteps:
                    loss_dict[f"{var}_{level}_idx_{time}"] = torch.sqrt(
                        w_atmos_error[:, time, i, l].mean()
                    )

        return loss_dict
