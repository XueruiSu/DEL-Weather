from functools import lru_cache
from typing import List

import numpy as np
import torch
from climai_global.modules.aggregation import EncoderAggregationBlock
from climai_global.modules.groupedpatchembed import StableGroupedVarPatchEmbed
from climai_global.modules.vit_block import TransformerEncoderBlock
from climai_global.modules.pos_encode import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_lat_lon_embed,
)
from timm.models.vision_transformer import trunc_normal_
from torch import nn

CONSTANTS = [
    "land_sea_mask",
    "orography",
    "lattitude",
]

SINGLE_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_cloud_cover",
    "total_column_water_vapour",
    "toa_incident_solar_radiation",
    "total_precipitation",
]

ATMOS_LEVELS = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]

ATMOS_VARS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    "relative_humidity",
]

DEFAULT_VARS = CONSTANTS + SINGLE_VARS + [f"{v}_{l}" for v in ATMOS_VARS for l in ATMOS_LEVELS]


class ClimaXv2(nn.Module):
    def __init__(
        self,
        const_vars: List[str] = CONSTANTS,
        single_vars: List[str] = SINGLE_VARS,
        atmos_vars: List[str] = ATMOS_VARS,
        atmos_levels: List[int] = ATMOS_LEVELS,
        patch_size: int = 4,
        embed_dim: int = 1024,
        depth: int = 8,
        decoder_depth: int = 2,
        num_heads: int = 16,
        mlp_ratio: float = 48 / 11,
        drop_path: float = 0.1,
        drop_rate: float = 0.1,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.const_vars = const_vars
        self.single_vars = single_vars
        self.atmos_vars = atmos_vars
        self.atmos_levels = atmos_levels

        self.all_vars = single_vars + [f"{v}_{l}" for v in atmos_vars for l in atmos_levels]

        self.token_embeds = StableGroupedVarPatchEmbed(
            max_vars=len(self.all_vars), patch_size=patch_size, embed_dim=embed_dim
        )
        # potentially add const_vars to positional encoding
        self.pos_embed_net = StableGroupedVarPatchEmbed(
            max_vars=1, patch_size=patch_size, embed_dim=1
        )

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables
        self.single_var_embed, self.single_var_map = self.create_1d_embedding(
            embed_dim, self.single_vars
        )
        self.atmos_var_embed, self.atmos_var_map = self.create_1d_embedding(
            embed_dim, self.atmos_vars
        )
        self.atmos_lev_embed, self.atmos_lev_map = self.create_1d_embedding(
            embed_dim, self.atmos_levels
        )
        _, self.all_vars_map = self.create_1d_embedding(embed_dim, self.all_vars)

        # learnable atmospheric pressure level and variable aggregation
        self.level_agg = EncoderAggregationBlock(
            input_dim=self.embed_dim,
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            num_output_tokens=1,
            add_mlp=False,
            use_flash_attn=use_flash_attn,
        )
        self.variable_agg = EncoderAggregationBlock(
            input_dim=self.embed_dim,
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            num_output_tokens=1,
            add_mlp=False,
            use_flash_attn=use_flash_attn,
        )

        # lead time embedding
        self.lead_time_embed = nn.Linear(1, embed_dim)

        # ViT backbone
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    drop=drop_rate,
                    use_flash_attn=use_flash_attn,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # prediction head
        head = nn.ModuleList()
        for _ in range(decoder_depth):
            head.append(nn.Linear(embed_dim, embed_dim))
            head.append(nn.GELU())
        head.append(nn.Linear(embed_dim, len(self.all_vars) * patch_size**2))
        self.head = nn.Sequential(*head)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize var_emb

        # single variable embedding, atmospheric pressure / variable embedding
        single_var_embed = get_1d_sincos_pos_embed_from_grid(
            self.single_var_embed.shape[-1], np.arange(len(self.single_var_embed))
        )
        self.single_var_embed.data.copy_(torch.from_numpy(single_var_embed).float())
        atmos_var_embed = get_1d_sincos_pos_embed_from_grid(
            self.atmos_var_embed.shape[-1], np.arange(len(self.atmos_var_embed))
        )
        self.atmos_var_embed.data.copy_(torch.from_numpy(atmos_var_embed).float())
        atmos_lev_embed = get_1d_sincos_pos_embed_from_grid(
            self.atmos_lev_embed.shape[-1], np.arange(len(self.atmos_lev_embed))
        )
        self.atmos_lev_embed.data.copy_(torch.from_numpy(atmos_lev_embed).float())

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_1d_embedding(self, dim, variables):
        var_embed = nn.Parameter(torch.zeros(len(variables), dim), requires_grad=True)
        # TODO: create a mapping from var --> idx
        var_map = {}
        idx = 0
        for var in variables:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_single_var_ids(self, vars, device):
        ids = torch.tensor([self.single_var_map[var] for var in vars], device=device)
        return ids

    @lru_cache(maxsize=None)
    def get_atmos_var_ids(self, vars, device):
        ids = torch.tensor([self.atmos_var_map[var] for var in vars], device=device)
        return ids

    @lru_cache(maxsize=None)
    def get_atmos_lev_ids(self, levels, device):
        ids = torch.tensor([self.atmos_lev_map[lev] for lev in levels], device=device)
        return ids

    @lru_cache(maxsize=None)
    def get_all_var_ids(self, all_vars, device):
        ids = torch.tensor([self.all_vars_map[var] for var in all_vars], device=device)
        return ids

    def get_single_var_emb(self, single_var_embed, single_vars):
        ids = self.get_single_var_ids(single_vars, single_var_embed.device)
        return single_var_embed[ids, :]

    def get_atmos_var_emb(self, atmos_var_embed, atmos_vars):
        ids = self.get_atmos_var_ids(atmos_vars, atmos_var_embed.device)
        return atmos_var_embed[ids, :]

    def get_atmos_lev_emb(self, atmos_lev_embed, atmos_levels):
        ids = self.get_atmos_lev_ids(atmos_levels, atmos_lev_embed.device)
        return atmos_lev_embed[ids, :]

    def unpatchify(self, x: torch.Tensor, h, w):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = len(self.all_vars)
        h = h // p
        w = w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def aggregate_levels(self, x: torch.Tensor):
        """
        x: (BxT, V_A, C_A, L, D) -> (BxT, V_A, hid, L, D)
        for hid = 1, this is a standard attention aggregation
        """
        b, v, _, l, _ = x.shape
        x = torch.einsum("bvcld->bvlcd", x)
        x = x.flatten(0, 2)  # BxTxV_AxL, C_A, D

        x = self.level_agg(x)  # BxTxV_AxL, hid, D
        x = x.unflatten(dim=0, sizes=(b, v, l))  # BxT, V_A, L, hid, D
        # TODO(jkg): Why do we rearrange the dimensions here?
        x = torch.einsum("bvlcd->bvcld", x)  # BxT, V_A, hid, L, D
        return x

    def aggregate_variables(self, x: torch.Tensor):
        """
        x: (BxT, V_A*hid + V_S, L, D) -> (BxT, 1, L, D)
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        x = self.variable_agg(x)  # BxL, 1, D
        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, 1, D
        return x

    def forward_encoder(
        self,
        x_single: torch.Tensor,
        x_atmos: torch.Tensor,
        lead_times: torch.Tensor,
        metadata,
    ):
        """
        x_single: `[B, T, V_S, H, W]`
        x_atmos: `[B, T, V_A, C_A, H, W]`
        """

        single_vars = tuple(metadata.single_vars)
        atmos_vars = tuple(metadata.atmos_vars.keys())
        atmos_levels = tuple(metadata.atmos_vars[atmos_vars[0]])
        all_vars = single_vars + tuple(
            f"{v}_{l}" for v in metadata.atmos_vars for l in metadata.atmos_vars[v]
        )

        x_single = x_single.flatten(0, 1)  # BxT, V_S, H, W
        x_atmos = x_atmos.flatten(0, 1)  # BxT, V_A, C_A, H, W
        _, VA, CA, H, W = x_atmos.size()
        assert len(metadata.lat) == H
        assert len(metadata.lon) == W

        # tokenize single and atmospheric variables
        # Note: we need ids for all varaibles as a separate token embedding layer is used for each variable
        # TODO: should we share the token embedding across different pressure levels? Likely not.
        # (BxT, V_S, H, W) -> (BxT, V_S, L, D)
        # (BxT, V_A, C_A, H, W) -> (BxT, V_A, C_A, L, D)
        var_ids = self.get_all_var_ids(all_vars, x_single.device)
        x_atmos = x_atmos.flatten(1, 2)
        # for the tokenization we add all variables together, such that we are safe with id matchings
        x = torch.cat((x_single, x_atmos), dim=1)
        x = self.token_embeds(x, var_ids)  # BxT, V, L, D
        x_single = x[:, : len(single_vars)]  # BxT, V_S, L, D
        x_atmos = x[:, len(single_vars) :].unflatten(dim=1, sizes=(VA, CA))  # BxT, V_A, C_A, L, D

        # add atmospheric pressure embedding
        # aggregate over pressure levels
        # (BxT, V_A, C_A, L, D) -> (BxT, V_A, hid, L, D)
        # for hid = 1, this is a standard attention aggregation
        atmos_levels_embed = self.get_atmos_lev_emb(self.atmos_lev_embed, atmos_levels)[
            None, None, :, None, :
        ]
        x_atmos = x_atmos + atmos_levels_embed
        x_atmos = self.aggregate_levels(x_atmos)

        # add single and atmospheric embedding
        # aggregate over variables
        # (BxT, V_A * hid + V_S, L, D) -> (BxT, 1, L, D)
        single_var_embed = self.get_single_var_emb(self.single_var_embed, single_vars)
        x_single = x_single + single_var_embed[None, :, None, :]
        atmos_var_embed = self.get_atmos_var_emb(self.atmos_var_embed, atmos_vars)
        x_atmos = x_atmos + atmos_var_embed[None, :, None, None, :]
        # flatten atmos variable and level dimension
        x = torch.cat((x_single, x_atmos.flatten(1, 2)), dim=1)
        x = self.aggregate_variables(x).squeeze()

        # add position and time embeddings
        # TODO: add e.g. orographic information here
        # we apply the same kernel to all D of the pos embedding which has the shape (H*W, D)
        pos_embed = get_2d_sincos_lat_lon_embed(
            self.embed_dim, metadata.lat, metadata.lon, x.device
        ).to(dtype=x.dtype)
        pos_embed = self.pos_embed_net(pos_embed.reshape(1, H, W, -1).permute(3, 0, 1, 2), vars=[0])
        # [hid, 1, D, 1] -> [D, hid]
        pos_embed = pos_embed.squeeze().permute(1, 0)

        # add pos embedding
        x = x + pos_embed[None, :]

        # add lead time embedding
        lead_times = lead_times.to(x.dtype)
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        lead_time_emb = lead_time_emb.unsqueeze(1)
        x = x + lead_time_emb  # B, L, D

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(
        self,
        single_inputs,
        atmos_inputs,
        lead_times,
        metadata,
        single_targets=None,
        atmos_targets=None,
        metric=None,
    ):
        """Forward pass through the model.

        Args:

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        variables = list(metadata.single_vars) + [
            f"{v}_{l}" for v in metadata.atmos_vars for l in metadata.atmos_vars[v]
        ]
        num_single_vars = len(metadata.single_vars)

        if single_targets is None and atmos_targets is None:
            if metric is not None:
                raise ValueError("metric must be None when no targets are provided")

        # B, T, L, H, W for each atmos variable
        atmos_level_lens = [atmos_inputs[k].shape[2] for k in atmos_inputs.keys()]
        # get the eventual indices of the atmos variables when flattened
        np.cumsum(atmos_level_lens)
        # TODO: fixme
        atmos_inputs = torch.stack([atmos_inputs[k] for k in atmos_inputs.keys()], dim=2)
        out_transformers = self.forward_encoder(
            single_inputs,
            atmos_inputs,
            lead_times,
            metadata,
        )  # B, L, D
        predtokens = self.head(out_transformers)  # B, L, (V_S + V_A*C_A)*p*p
        H, W = len(metadata.lat), len(metadata.lon)
        out_var_ids = self.get_all_var_ids(tuple(variables), predtokens.device)
        preds = self.unpatchify(predtokens, H, W)[:, out_var_ids]
        single_preds = preds[:, :num_single_vars]
        atmos_preds = preds[:, num_single_vars:].reshape(
            preds.shape[0],
            len(metadata.atmos_vars),
            -1,
            H,
            W,
        )
        # TODO: fixme
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

    def evaluate(
        self,
        single_inputs,
        atmos_inputs,
        single_targets,
        atmos_targets,
        lead_times,
        metadata,
        metrics,
    ):
        with torch.no_grad():
            _, single_preds, atmos_preds = self.forward(
                single_inputs,
                atmos_inputs,
                lead_times,
                metadata,
                single_targets,
                atmos_targets,
                metrics,
            )
        loss = [
            m(single_preds, single_targets, atmos_preds, atmos_targets, metadata) for m in metrics
        ]
        return loss, single_preds, atmos_preds
