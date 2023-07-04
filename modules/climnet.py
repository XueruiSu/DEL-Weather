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


class Down_sample_Vit(nn.Module):
    def __init__(
        self,
        decoder_depth_down: int = 2,
        embed_dim: int = 1024,
        out_embed_dim: int = 512,
        depth: int = 2,
        num_heads: int = 16,
        mlp_ratio: float = 48 / 11,
        drop_path: float = 0.1,
        drop_rate: float = 0.1,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.Vit = TransformerEncoderBackbone(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            drop_rate=drop_rate,
            use_flash_attn=use_flash_attn,
        )
        head = nn.ModuleList()
        for _ in range(decoder_depth_down):
            head.append(nn.Linear(embed_dim, embed_dim))
            head.append(nn.GELU())
        head.append(nn.Linear(embed_dim, out_embed_dim))
        self.down_sample = nn.Sequential(*head)
    
    def forward(self, x):
        x = self.Vit(x)
        x = self.down_sample(x)
        return x
        
class decoder_UQ(nn.Module):
    def __init__(
        self,
        embed_dim_UQ: int = 768,
        out_embed_dim_UQ = [512, 191],
        en_UQ_embed_dim = [384, 256],
        depth_UQ: int = 2,
        num_heads_UQ_down: int = 4,
        mlp_ratio_UQ: float = 48 / 11,
        drop_path_UQ: float = 0.1,
        drop_rate_UQ: float = 0.1,
        use_flash_attn_UQ: bool = False,
        decoder_depth_down_UQ: int = 1,
    ):
        super().__init__()
        
        self.Down_sample_Vit1 = Down_sample_Vit(
            decoder_depth_down=decoder_depth_down_UQ,
            embed_dim=embed_dim_UQ,
            out_embed_dim=out_embed_dim_UQ[0],
            depth=depth_UQ,
            num_heads=num_heads_UQ_down,
            mlp_ratio=mlp_ratio_UQ,
            drop_path=drop_path_UQ,
            drop_rate=drop_rate_UQ,
            use_flash_attn=use_flash_attn_UQ,
        )
        self.Down_sample_Vit2 = Down_sample_Vit(
            decoder_depth_down=decoder_depth_down_UQ,
            embed_dim=out_embed_dim_UQ[0],
            out_embed_dim=out_embed_dim_UQ[1],
            depth=depth_UQ,
            num_heads=num_heads_UQ_down,
            mlp_ratio=mlp_ratio_UQ,
            drop_path=drop_path_UQ,
            drop_rate=drop_rate_UQ,
            use_flash_attn=use_flash_attn_UQ,
        )
        self.decoder = nn.Sequential(  
            nn.Linear(4050, 1024),   
            nn.GELU(),  
            nn.Linear(1024, 256),  
            nn.GELU(),  
            nn.Linear(256, 1),   
        )  
        self.En2Down_1 = nn.Linear(en_UQ_embed_dim[0], out_embed_dim_UQ[0])
        self.En2Down_2 = nn.Linear(en_UQ_embed_dim[1], out_embed_dim_UQ[1])
        
    def forward(self, x, x1, x2):
        # x: (B, 4050, 768), x1: (B, 4050, en_UQ_embed_dim[0]), x2: (B, 4050, en_UQ_embed_dim[1])
        # self.Down_sample_Vit1(x): (B, 4050, out_embed_dim_UQ[0])
        # self.Down_sample_Vit2(x): (B, 4050, out_embed_dim_UQ[1])
        x = self.Down_sample_Vit2(self.Down_sample_Vit1(x) + self.En2Down_1(x1)) + self.En2Down_2(x2) # (B, 4050, 191)
        R2Loss_pre = self.decoder(x.permute(0, 2, 1)).squeeze()
        return R2Loss_pre

class ClimNet_UQ(nn.Module):
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
        
        embed_dim_UQ: int = 768,
        out_embed_dim_UQ = [512, 191],
        depth_UQ: int = 2,
        en_UQ_embed_dim = [256, 256],
        num_heads_UQ_en: int = 2,
        num_heads_UQ_down: int = 4,
        mlp_ratio_UQ: float = 48 / 11,
        drop_path_UQ: float = 0.1,
        drop_rate_UQ: float = 0.1,
        use_flash_attn_UQ: bool = False,
        decoder_depth_down_UQ: int = 1,
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
        self.backbone = TransformerEncoderBackbone(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            drop_rate=drop_rate,
            use_flash_attn=use_flash_attn,
        )     
        
        self.encoder_hat1 = MultiScaleEncoder(
            single_vars=single_vars,
            atmos_vars=atmos_vars,
            atmos_levels=atmos_levels,
            patch_size=patch_size,
            embed_dim=en_UQ_embed_dim[0],
            num_heads=num_heads_UQ_en,
            pressure_encode_dim=pressure_encode_dim,
            drop_rate=drop_rate,
        )
        self.encoder_hat2 = MultiScaleEncoder(
            single_vars=single_vars,
            atmos_vars=atmos_vars,
            atmos_levels=atmos_levels,
            patch_size=patch_size,
            embed_dim=en_UQ_embed_dim[1],
            num_heads=num_heads_UQ_en,
            pressure_encode_dim=pressure_encode_dim,
            drop_rate=drop_rate,
        )
        
        self.decoder = decoder_UQ(
            embed_dim_UQ = embed_dim_UQ,
            out_embed_dim_UQ = out_embed_dim_UQ,
            en_UQ_embed_dim = en_UQ_embed_dim,
            depth_UQ = depth_UQ,
            num_heads_UQ_down = num_heads_UQ_down,
            mlp_ratio_UQ = mlp_ratio_UQ,
            drop_path_UQ = drop_path_UQ,
            drop_rate_UQ = drop_rate_UQ,
            use_flash_attn_UQ = use_flash_attn_UQ,
            decoder_depth_down_UQ = decoder_depth_down_UQ,
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
        single_preds: torch.Tensor, 
        atmos_preds: dict[str, torch.Tensor],
        R2Loss_target: Optional[torch.Tensor] = None,
        metric=None,
    ) -> tuple[Optional[list], torch.Tensor]:
        
        if R2Loss_target is None:
            if metric is not None:
                raise ValueError("metric must be None when no targets are provided")
        atmos_inputs = torch.stack([atmos_inputs[k] for k in atmos_inputs.keys()], dim=2)
        x = self.encoder(single_inputs, atmos_inputs, lead_times, metadata)
        x = self.backbone(x) # (B, 4050, 768)
        
        # u_t+1_hat postprogress:
        if single_preds.shape[1] != 1:
            single_preds = single_preds.unsqueeze(1)
        if atmos_preds[list(atmos_preds.keys())[0]].shape[1] != 1:
            for k in atmos_preds.keys():
                atmos_preds[k] = atmos_preds[k].unsqueeze(1)
        atmos_preds = torch.stack([atmos_preds[k] for k in atmos_preds.keys()], dim=2)
        x1 = self.encoder_hat1(single_preds, atmos_preds, lead_times, metadata) # (B, 4050, 256)
        x2 = self.encoder_hat2(single_preds, atmos_preds, lead_times, metadata) # (B, 4050, 191)
        
        # UQ inference
        R2Loss_pre = self.decoder(x, x1, x2)
        
        if metric is None:
            loss = None
        else:
            loss = [m(R2Loss_pre, R2Loss_target) for m in metric]

        return loss, R2Loss_pre

    # TODO (Xuerui): need to check metrics.
    @torch.no_grad()
    def evaluate(
        self,
        single_inputs,
        atmos_inputs,
        lead_times,
        metadata,
        R2Loss_target,
        metrics,
    ):
        _, R2Loss_pre = self.forward(
            single_inputs, atmos_inputs, lead_times, metadata, R2Loss_target, None
        )

        # calculate metrics
        loss = [
            m(R2Loss_pre, R2Loss_target) for m in metrics
        ]

        return loss, (
            R2Loss_pre,
            R2Loss_target,
            lead_times,
            metadata,
        )

