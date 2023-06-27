import torch
from torch import nn
from typing import Optional
from data import Metadata
from timm.models.vision_transformer import trunc_normal_

from modules.vit_block import TransformerEncoderBackbone
from modules.multiscaleblocks import MultiScaleEncoder
from modules.multiscaleblocks import MultiScaleDecoder

CONSTANTS = ("",)
SINGLE_VARS = ("tas",)
ATMOS_LEVELS = (
    50,
    250,
    500,
    600,
    700,
    850,
)
ATMOS_VARS = ("ta",)


class ClimNet(nn.Module):
    def __init__(
        self,
        const_vars: tuple[str, ...] = CONSTANTS,
        single_vars: tuple[str, ...] = SINGLE_VARS,
        atmos_vars: tuple[str, ...] = ATMOS_VARS,
        atmos_levels: tuple[int, ...] = ATMOS_LEVELS,
        patch_size: int = 4,
        embed_dim: int = 1024,
        pressure_encode_dim: int = 1,
        depth: int = 8,
        decoder_depth: int = 2,
        num_heads: int = 16,
        mlp_ratio: float = 48 / 11,
        drop_path: float = 0.1,
        drop_rate: float = 0.1,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.const_vars = const_vars

        self.encoder = MultiScaleEncoder(
            single_vars=single_vars,
            atmos_vars=atmos_vars,
            atmos_levels=atmos_levels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            pressure_encode_dim=pressure_encode_dim,
            drop_rate=drop_rate,
        )
        # TODO (Johannes): add StaticEncoder
        # self.static_encoder = StaticEncoder(
        #     const_vars = const_vars,
        #     patch_size = patch_size,
        #     embed_dim = embed_dim,
        # )
        self.backbone = TransformerEncoderBackbone(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            drop_rate=drop_rate,
            use_flash_attn=use_flash_attn,
        )
        self.decoder = MultiScaleDecoder(
            single_vars=single_vars,
            atmos_vars=atmos_vars,
            atmos_levels=atmos_levels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            decoder_depth=decoder_depth,
        )

        # initialize nn.Linear and nn.LayerNorm
        # TODO(Johannes): not sure if this is the right weight initialization -> initial outputs have rather small values
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        single_inputs: torch.Tensor,
        atmos_inputs: dict[str, torch.Tensor],
        lead_times: torch.Tensor,
        metadata: Metadata,
        single_targets: Optional[torch.Tensor] = None,
        atmos_targets: Optional[dict[str, torch.Tensor]] = None,
        metric=None,
    ) -> tuple[Optional[list], torch.Tensor, dict]:
        """Forward pass of the ClimNet model.

        Args:
            single_inputs (torch.Tensor): `(batch=B, time=T, single_vars=V_S, height(latitute)=H, width(longitude)=W)`
            atmos_inputs (dict[str, torch.Tensor]): Dict of atmos variables where each entry is of `(batch=B, time=T, atmos_levels=C_A, height(latitute)=H, width(longitude)=W)`
            lead_times (torch.Tensor): `(batch=B)`
            metadata (Metadata): Metadata information.
            single_targets (torch.Tensor, optional): `(B, V_S, H, W)`. Defaults to None.
            atmos_targets (dict[str, torch.Tensor], optional): Dict of atmos variables where each entry is of `(batch=B, time=T, atmos_levels=C_A, height(latitute)=H, width(longitude)=W)`.
            Defaults to None.
            metric: Metrics. Defaults to None.

        Raises:
            ValueError: Throws when no metric is provided.

        Returns:
            tuple[Optional[list], torch.Tensor, dict]: Loss tuple for different metrics, single var predictions of `(B, V_S, H, W)`,
            Single variable predictions of `(batch=B, time=T, single_vars=V_S, height(latitute)=H, width(longitude)=W)`
            Dict of atmos variable predictions where each entry is of `(batch=B, time=T, atmos_levels=C_A, height(latitute)=H, width(longitude)=W)`
        """
        if single_targets is None and atmos_targets is None:
            if metric is not None:
                raise ValueError("metric must be None when no targets are provided")

        atmos_inputs = torch.stack([atmos_inputs[k] for k in atmos_inputs.keys()], dim=2)
        x = self.encoder(single_inputs, atmos_inputs, lead_times, metadata)
        x = self.backbone(x)
        single_preds, atmos_preds = self.decoder(x, metadata)

        # TODO (Johannes): Tests for this part of the code.
        atmos_outputs = {}
        for idx, k in enumerate(metadata.atmos_vars.keys()):
            atmos_outputs[k] = atmos_preds[:, idx]

        if metric is None:
            loss = None
        else:
            loss = [
                m(single_preds, single_targets, atmos_outputs, atmos_targets, metadata)
                for m in metric
            ]

        return loss, single_preds, atmos_outputs

    # TODO (Johannes): need to structure metrics a bit better.
    @torch.no_grad()
    def evaluate(
        self,
        single_inputs,
        atmos_inputs,
        lead_times,
        metadata,
        single_targets,
        atmos_targets,
        metrics,
    ):
        """Evaluate pass of the ClimNet model.

        It returns all the infomation that describes the prediction and the metrics.

        Args:
            single_inputs (torch.Tensor): `(batch=B, time=T, single_vars=V_S, height(latitute)=H, width(longitude)=W)`
            atmos_inputs (dict[str, torch.Tensor]): Dict of atmos variables where each entry is of `(batch=B, time=T, atmos_levels=C_A, height(latitute)=H, width(longitude)=W)`
            lead_times (torch.Tensor): `(batch=B)`
            metadata (Metadata): Metadata information.
            single_targets (torch.Tensor, optional): `(B, V_S, H, W)`. Defaults to None.
            atmos_targets (dict[str, torch.Tensor], optional): Dict of atmos variables where each entry is of `(batch=B, time=T, atmos_levels=C_A, height(latitute)=H, width(longitude)=W)`.
            Defaults to None.
            metric: Metrics. Defaults to None.

        Returns:
            tuple[Optional[list], torch.Tensor, dict]: Loss tuple for different metrics, single var predictions of `(B, V_S, H, W)`,
            Single variable predictions of `(batch=B, time=T, single_vars=V_S, height(latitute)=H, width(longitude)=W)`
            Single variable target of `(batch=B, time=T, single_vars=V_S, height(latitute)=H, width(longitude)=W)`
            Dict of atmos variable predictions where each entry is of `(batch=B, time=T, atmos_levels=C_A, height(latitute)=H, width(longitude)=W)`
            Dict of atmos variable targets where each entry is of `(batch=B, time=T, atmos_levels=C_A, height(latitute)=H, width(longitude)=W)`
            lead_times (torch.Tensor): `(batch=B)`
            metadata (Metadata): Metadata information.
        """
        # prediction
        _, single_preds, atmos_preds = self.forward(
            single_inputs, atmos_inputs, lead_times, metadata, single_targets, atmos_targets, None
        )

        # calculate metrics
        loss = [
            m(single_preds, single_targets, atmos_preds, atmos_targets, metadata) for m in metrics
        ]

        return loss, (
            single_preds,
            single_targets,
            atmos_preds,
            atmos_targets,
            lead_times,
            metadata,
        )
