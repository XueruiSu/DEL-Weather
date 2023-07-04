import torch
from torch import nn
from data import Metadata
from modules.pos_encode import (
    get_1d_pos_encode,
    get_2d_patched_lat_lon_encode,
)
from modules.aggregation import EncoderAggregationBlock
from modules.groupedpatchembed import StableGroupedVarPatchEmbed


def create_var_map(vars: tuple) -> dict:
    """Create variable dictionary where the keys are the variable names and queries are unique assigned ids.

    Args:
        vars (tuple): Variable strings.

    Returns:
        dict: Variable map dictionary.
    """
    var_map = {}
    idx = 0
    for var in vars:
        var_map[var] = idx
        idx += 1
    return var_map


def get_ids_for_var_map(vars: tuple, var_maps: dict, device: torch.cuda.device) -> torch.Tensor:
    """Returns tensor of variable ids after retrieving those from a "variable"/id dictionary.

    Args:
        vars (tuple): Look up variables.
        var_maps (dict): Variable dictionary used to extract ids
        device (torch.cuda.device): device

    Returns:
        torch.Tensor: Tensor of variable ids found in var_maps dictionary.
    """
    ids = torch.tensor([var_maps[var] for var in vars], device=device)
    return ids


class MultiScaleEncoder(nn.Module):
    """Multi-scale multi-source multi-variable encoder which step-wise aggregates
    atmospheric (pressure levels), variable (surface, atmospheric) and grid information (lat/lon position, scale, time).
    Variable independent patch encoding is used.

    Args:
        single_vars (tuple[str, ...]): Tuple of single variables.
        atmos_vars (tuple[str, ...]): Tuple of atmospheric variables.
        atmos_levels (tuple[int, ...]): Tuple of pressure levels.
        patch_size (int, optional): Patch size. Defaults to 4.
        embed_dim (int, optional): Embedding dim as used in the aggregation blocks and as encoding dim for position/scale embedding. Defaults to 1024.
        num_heads (int, optional): Number of attention heads used in aggregation blocks. Defaults to 16.
        pressure_encode_dim (int, optional): Dimension of pressure aggregation, pressure_encode_dim=1 equals attention based pooling over pressure dims. Defaults to 1.
        drop_rate (float, optional): Drop out rate for input patches. Defaults to 0.1.
    """

    def __init__(
        self,
        single_vars: tuple[str, ...],
        atmos_vars: tuple[str, ...],
        atmos_levels: tuple[int, ...],
        patch_size: int = 4,
        embed_dim: int = 1024,
        num_heads: int = 16,
        pressure_encode_dim: int = 1,
        drop_rate: float = 0.1,
    ):
        super().__init__()

        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.all_vars = single_vars + tuple(f"{v}_{l}" for v in atmos_vars for l in atmos_levels)

        # variable maps for all encodings
        self.all_var_map = create_var_map(self.all_vars)
        self.single_var_map = create_var_map(single_vars)
        self.atmos_var_map = create_var_map(atmos_vars)
        self.atmos_lev_map = create_var_map(atmos_levels)

        # embeddings of surface (single) variables, atmospheric variables, pressure levels
        # embed_dim is used as encoding dimension
        # positional encodings are made persistent to bind variable and pressure encodings to the architecture
        single_var_encode = get_1d_pos_encode(embed_dim, torch.arange(len(single_vars)))
        self.register_buffer("single_var_encode", single_var_encode, persistent=True)
        atmos_var_encode = get_1d_pos_encode(embed_dim, torch.arange(len(atmos_vars)))
        self.register_buffer("atmos_var_encode", atmos_var_encode, persistent=True)
        atmos_lev_encode = get_1d_pos_encode(embed_dim, torch.arange(len(atmos_levels)))
        self.register_buffer("atmos_lev_encode", atmos_lev_encode, persistent=True)

        # lead time embedding
        self.lead_time_embed = nn.Linear(1, embed_dim)

        # token embeddings
        self.token_embeds = StableGroupedVarPatchEmbed(len(self.all_vars), patch_size, embed_dim)

        # learnable atmospheric pressure level and variable aggregation
        self.level_agg = EncoderAggregationBlock(
            input_dim=embed_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_output_tokens=pressure_encode_dim,
            add_mlp=False,
        )
        # variable aggregation currently aggregates to 1 to be concise with ViT inputs.
        self.variable_agg = EncoderAggregationBlock(
            input_dim=embed_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_output_tokens=1,
            add_mlp=False,
        )

        # drop patches after encoding
        self.pos_drop = nn.Dropout(p=drop_rate)

    def filter_single_var_encoding_by_id(
        self, single_var_encode: torch.Tensor, single_vars: tuple[str, ...]
    ) -> torch.Tensor:
        """Filter single var encodings by var strings.

        Args:
            single_var_encode (torch.Tensor): Encoding tensor of single vars.
            single_vars (tuple[string, ...]): Single var strings for which encodings are needed.

        Returns:
            torch.Tensor: Tensor of single var encodings.
        """
        ids = get_ids_for_var_map(single_vars, self.single_var_map, single_var_encode.device)
        return single_var_encode[ids, :]

    def filter_atmos_var_encoding_by_id(
        self, atmos_var_encode: torch.Tensor, atmos_vars: tuple[str, ...]
    ) -> torch.Tensor:
        """Filter atmos var encodings by var strings.

        Args:
            atmos_var_encode (torch.Tensor): Encoding tensor of atmos vars.
            atmos_vars (tuple[string, ...]): Atmos var strings for which encodings are needed.

        Returns:
            torch.Tensor: Tensor of atmos var encodings.
        """
        ids = get_ids_for_var_map(atmos_vars, self.atmos_var_map, atmos_var_encode.device)
        return atmos_var_encode[ids, :]

    def filter_atmos_lev_encoding_by_id(
        self, atmos_lev_encode: torch.Tensor, atmos_levels: tuple[int, ...]
    ) -> torch.Tensor:
        """Filter atmos lev encodings by atmospheric levels (int).

        Args:
            atmos_lev_encode (torch.Tensor): Encoding tensor of atmos levels.
            atmos_levels (tuple[int, ...]): Atmos levels for which encodings are needed.

        Returns:
            torch.Tensor: Tensor of atmos level encodings.
        """
        ids = get_ids_for_var_map(atmos_levels, self.atmos_lev_map, atmos_lev_encode.device)
        return atmos_lev_encode[ids, :]

    def aggregate_levels(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate pressure level information. For hid = 1, it defaults to attention aggregation.

        Args:
            x (torch.Tensor): `(BxT, V_A, C_A, L, D)`

        Returns:
            torch.Tensor: `(BxT, V_A, hid, L, D)`
        """
        b, v, _, l, _ = x.shape
        x = torch.einsum("bvcld->bvlcd", x)
        x = x.flatten(0, 2)  # (BxTxV_AxL, C_A, D)

        x = self.level_agg(x)  # (BxTxV_AxL, hid, D)
        x = x.unflatten(dim=0, sizes=(b, v, l))  # (BxT, V_A, L, hid, D)
        x = torch.einsum("bvlcd->bvcld", x)  # (BxT, V_A, hid, L, D)
        return x

    def aggregate_variables(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate variable information. For hid = 1, it defaults to attention aggregation.

        Args:
            x (torch.Tensor): `(BxT, V_A*hid + V_S, L, D)`

        Returns:
            torch.Tensor: `(BxT, 1, L, D)`
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # (BxL, V, D)

        x = self.variable_agg(x)  # (BxL, 1, D)
        x = x.unflatten(dim=0, sizes=(b, l))  # (B, L, 1, D)
        return x

    def forward(
        self,
        x_single: torch.Tensor,
        x_atmos: torch.Tensor,
        lead_times: torch.Tensor,
        metadata: Metadata,
    ) -> torch.Tensor:
        """Forward pass of MultiScaleEncoder.

        Args:
            x_single (torch.Tensor): `(batch=B, time=T, single_vars=V_S, height(latitute)=H, width(longitude)=W)`
            x_atmos (torch.Tensor): `(batch=B, time=T, atmos_vars=V_A, atmos_levels=C_A, height(latitute)=H, width(longitude)=W)`
            lead_times (torch.Tensor): `(batch=B)`
            metadata (Metadata): Metadata information.

        Returns:
            torch.Tensor: `(B, L=number_of_tokens, D=embed_dim)`
        """
        single_vars = tuple(metadata.single_vars)
        atmos_vars = tuple(metadata.atmos_vars.keys())
        atmos_levels = tuple(metadata.atmos_vars[atmos_vars[0]])
        # Keep lat and lon the same dtype of x.
        lat, lon = metadata.lat.to(x_single.dtype), metadata.lon.to(x_single.dtype)
        all_vars = single_vars + tuple(
            f"{v}_{l}" for v in metadata.atmos_vars for l in metadata.atmos_vars[v]
        )

        x_single = x_single.flatten(0, 1)  # (BxT, V_S, H, W)
        x_atmos = x_atmos.flatten(0, 1)  # (BxT, V_A, C_A, H, W)
        _, VA, CA, H, W = x_atmos.size()
        assert len(lat) == H
        assert len(lon) == W

        # tokenize single and atmospheric variables
        # Note: we need ids for all varaibles as a separate token embedding layer is used for each variable
        # TODO: should we share the token embedding across different pressure levels? Likely not.
        # (BxT, V_S, H, W) -> (BxT, V_S, L, D)
        # (BxT, V_A, C_A, H, W) -> (BxT, V_A, C_A, L, D)
        all_var_ids = get_ids_for_var_map(all_vars, self.all_var_map, x_single.device)
        x_atmos = x_atmos.flatten(1, 2)
        # for the tokenization we add all variables together, such that we are safe with id matchings
        x = torch.cat((x_single, x_atmos), dim=1)
        # token embedding + id filtering of tokens
        x = self.token_embeds(x, all_var_ids)  # (BxT, V, L, D)
        x_single = x[:, : len(single_vars)]  # (BxT, V_S, L, D)
        x_atmos = x[:, len(single_vars) :].unflatten(dim=1, sizes=(VA, CA))  # (BxT, V_A, C_A, L, D)

        # add atmospheric pressure encoding + id filtering for pressure levels
        atmos_levels_encode = self.filter_atmos_lev_encoding_by_id(
            self.atmos_lev_encode, atmos_levels
        )[
            None, None, :, None, :
        ]  # (1, 1, C_A, 1, D)
        x_atmos = x_atmos + atmos_levels_encode
        # aggregate over pressure levels
        # (BxT, V_A, C_A, L, D) -> (BxT, V_A, hid, L, D)
        # for hid = 1, this is a standard attention aggregation
        x_atmos = self.aggregate_levels(x_atmos)

        # add single and atmospheric encoding + id filtering for single and pressure vars
        single_var_encode = self.filter_single_var_encoding_by_id(
            self.single_var_encode, single_vars
        )
        # (BxT, V_S, L, D) + (1, V_S, 1, D)
        x_single = x_single + single_var_encode[None, :, None, :]
        atmos_var_encode = self.filter_atmos_var_encoding_by_id(self.atmos_var_encode, atmos_vars)
        # (BxT, V_A, hid, L, D) + (1, V_A, 1, 1, D)
        x_atmos = x_atmos + atmos_var_encode[None, :, None, None, :]
        # flatten atmos variable and level dimension
        x = torch.cat((x_single, x_atmos.flatten(1, 2)), dim=1)
        # aggregate over variables
        # (BxT, V_A * hid + V_S, L, D) -> (BxT, 1, L, D)
        # TODO (Johannes): for num_out_tokens != 1, we need a mapping (BxT, hid2, L, D) -> (BxT, L, D)
        x = self.aggregate_variables(x).squeeze()

        # add position and scale embeddings
        pos_encode, scale_encode = get_2d_patched_lat_lon_encode(
            self.embed_dim, lat, lon, self.patch_size
        )
        # (BxT, L, D) + (1, L, D) + (1, L, D)
        x = x + pos_encode[None, :] + scale_encode[None, :]

        # add lead time embedding
        lead_times = lead_times.to(x.dtype)
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        lead_time_emb = lead_time_emb.unsqueeze(1)
        # (BxT, L, D) + (1, 1, D)
        x = x + lead_time_emb

        # TODO (Johannes): If we ever consider T > 1, this would be the place to collapse (BxT, L, D) -> (B, L, D)

        # drop patches
        x = self.pos_drop(x)

        return x


class MultiScaleDecoder(nn.Module):
    """Multi-scale multi-source multi-variable decoder which decodes back to
    atmospheric variables per pressure levels and single variables.

        Args:
            single_vars (tuple[str, ...]): Tuple of single variables.
            atmos_vars (tuple[str, ...]): Tuple of atmospheric variables.
            atmos_levels (tuple[int, ...]): Tuple of pressure levels.
            patch_size (int, optional): Patch size. Defaults to 4.
            embed_dim (int, optional): Embedding dim which is decoded. Defaults to 1024.
            decoder_depth (int, optional): Number of layers in the decoder head. Defaults to 2.
    """

    def __init__(
        self,
        single_vars: tuple[str, ...],
        atmos_vars: tuple[str, ...],
        atmos_levels: tuple[int, ...],
        patch_size: int = 4,
        embed_dim: int = 1024,
        decoder_depth: int = 2,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.all_vars = single_vars + tuple(f"{v}_{l}" for v in atmos_vars for l in atmos_levels)
        self.all_var_map = create_var_map(self.all_vars)

        # prediction head
        # TODO (Johannes): we probably want a normalization in the pedriction head as well
        head = nn.ModuleList()
        for _ in range(decoder_depth):
            head.append(nn.Linear(embed_dim, embed_dim))
            head.append(nn.GELU())
        head.append(nn.Linear(embed_dim, len(self.all_vars) * patch_size**2))
        self.head = nn.Sequential(*head)

    def unpatchify(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Unpatchify hidden representation.

        Args:
            x (torch.Tensor): `(B, L, V * patch_size**2)`
            H (int): latitude values
            W (int): longitude values

        Returns:
            torch.Tensor: `(B, V, H, W)`
        """
        P = self.patch_size
        C = len(self.all_vars)
        H = H // P
        W = W // P
        assert H * W == x.shape[1]

        x = x.reshape(shape=(x.shape[0], H, W, P, P, C))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x_unpatched = x.reshape(shape=(x.shape[0], C, H * P, W * P))
        return x_unpatched

    def forward(
        self,
        x: torch.Tensor,
        metadata: Metadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of MultiScaleEncoder.

        Args:
            x (torch.Tensor): `(B, L, D)`.
            metadata (Metadata): Metadata information.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: `(batch=B, time=T, single_vars=V_S, height(latitute)=H, width(longitude)=W)`
            and `(batch=B, time=T, atmos_vars=V_A, atmos_levels=C_A, height(latitute)=H, width(longitude)=W)`
        """
        all_vars = metadata.single_vars + tuple(
            f"{v}_{l}" for v in metadata.atmos_vars for l in metadata.atmos_vars[v]
        )
        num_single_vars = len(metadata.single_vars)
        H, W = len(metadata.lat), len(metadata.lon)

        x = self.head(x)  # B, L, (V_S + V_A*C_A)*p*p
        out_var_ids = get_ids_for_var_map(all_vars, self.all_var_map, x.device)
        preds = self.unpatchify(x, H, W)[:, out_var_ids]
        single_preds = preds[:, :num_single_vars]
        atmos_preds = preds[:, num_single_vars:].reshape(
            preds.shape[0],
            len(metadata.atmos_vars),
            -1,
            H,
            W,
        )

        return single_preds, atmos_preds
