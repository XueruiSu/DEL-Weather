from functools import lru_cache
from typing import List

import numpy as np
import torch
from climai_global.modules.groupedpatchembed import GroupedVarPatchEmbed
from climai_global.modules.vit_block import TransformerEncoderBlock
from climai_global.modules.pos_encode import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
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
    # "toa_incident_solar_radiation",
    # 'total_precipitation',
]

ATMOS_LEVELS = [50, 250, 500, 600, 700, 850, 925]
ATMOS_VARS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "relative_humidity",
    "specific_humidity",
]

DEFAULT_VARS = CONSTANTS + SINGLE_VARS + [f"{v}_{l}" for v in ATMOS_VARS for l in ATMOS_LEVELS]


class ClimaXv1(nn.Module):
    def __init__(
        self,
        default_vars: List[str] = DEFAULT_VARS,
        grid_size: List[int] = [128, 256],
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
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.default_vars = default_vars

        self.token_embeds = GroupedVarPatchEmbed(
            len(default_vars), grid_size, patch_size, embed_dim
        )
        self.num_patches = self.token_embeds.num_patches

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables
        self.var_embed, self.var_map = self.create_var_embedding(embed_dim)

        # variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # positional embedding and lead time embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=True
        )
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
        head.append(nn.Linear(embed_dim, len(self.default_vars) * patch_size**2))
        self.head = nn.Sequential(*head)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_emb and var_emb
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.grid_size[0] / self.patch_size),
            int(self.grid_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        var_embed = get_1d_sincos_pos_embed_from_grid(
            self.var_embed.shape[-1], np.arange(len(self.default_vars))
        )
        self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

        # initialize token_embeds
        for i in range(len(self.token_embeds.proj_weights)):
            w = self.token_embeds.proj_weights[i]
            trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

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

    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, len(self.default_vars), dim), requires_grad=True)
        # TODO: create a mapping from var --> idx
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = torch.tensor([self.var_map[var] for var in vars], device=device)
        return ids

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = len(self.default_vars)
        h = self.grid_size[0] // p if h is None else h // p
        w = self.grid_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.var_agg(var_query, x, x)  # BxL, D
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def forward_encoder(self, x: torch.Tensor, lead_times: torch.Tensor, variables):
        # x: `[B, T, V, H, W]` shape.

        if isinstance(variables, list):
            variables = tuple(variables)

        x = x.flatten(0, 1)  # BxT, V, H, W

        # tokenize each variable separately
        var_ids = self.get_var_ids(variables, x.device)
        x = self.token_embeds(x, var_ids)  # BxT, C, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)  # B, V, L, D

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        # add pos embedding
        x = x + self.pos_embed

        # add lead time embedding
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
        if single_targets is None or atmos_targets is None:
            if metric is not None:
                raise ValueError("metric must be None when no targets are provided")

        # B, T, L, H, W for each atmos variable
        atmos_level_lens = [atmos_inputs[k].shape[2] for k in atmos_inputs.keys()]
        # get the eventual indices of the atmos variables when flattened
        atmos_var_idxs = np.cumsum(atmos_level_lens)
        f_atmos_inputs = torch.cat([atmos_inputs[k] for k in atmos_inputs.keys()], dim=2)  # B, L, D
        if single_inputs is None:
            x = f_atmos_inputs
        else:
            x = torch.cat([single_inputs, f_atmos_inputs], dim=2)
        variables = list(metadata.single_vars) + [
            f"{v}_{l}" for v in metadata.atmos_vars for l in metadata.atmos_vars[v]
        ]

        out_transformers = self.forward_encoder(x, lead_times, variables)  # B, L, D
        predtokens = self.head(out_transformers)  # B, L, V*p*p

        num_single_vars = len(metadata.single_vars)
        out_var_ids = self.get_var_ids(tuple(variables), x.device)
        preds = self.unpatchify(predtokens)[:, out_var_ids]
        single_preds = preds[:, :num_single_vars]
        atmos_preds = preds[:, num_single_vars:]
        atmos_outputs = {}
        for idx, k in enumerate(atmos_inputs.keys()):
            # get the indices for the current atmos variable
            start_idx = 0 if idx == 0 else atmos_var_idxs[idx - 1]
            end_idx = atmos_var_idxs[idx]
            atmos_outputs[k] = atmos_preds[:, start_idx:end_idx]

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
        log_postfix,
    ):
        with torch.no_grad():
            _, single_preds, atmos_preds = self.forward(
                single_inputs,
                atmos_inputs,
                single_targets,
                atmos_targets,
                lead_times,
                metadata,
                metrics,
            )
        return [
            m(single_preds, single_targets, atmos_preds, atmos_targets, log_postfix)
            for m in metrics
        ]
