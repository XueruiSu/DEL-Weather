# TODO(Johannes): not sure what to do with the licensing, now all used functions have been newly written by myself
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Union
# from timm.layers.helpers import to_2tuple
from timm.models.layers.helpers import to_2tuple


def get_1d_pos_encode(
    encode_dim: int, pos: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Python version of 1d positional embedding.

    Args:
        encode_dim (int): Output encoding dimension `D`.
        pos (torch.Tensor): a list of positions to be encoded: size `(M,)`.
        dtype (torch.dtype, optional): Data type for omega parameter. Defaults to torch.float32.

    Returns:
        torch.Tensor: output tensor of dimensions `(M, D)`.
    """
    assert encode_dim % 2 == 0
    omega = torch.arange(encode_dim // 2, dtype=dtype, device=pos.device)
    omega /= encode_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    # (M, D/2), outer product
    out = torch.outer(pos, omega)

    encode_sin = torch.sin(out)  # (M, D/2)
    encode_cos = torch.cos(out)  # (M, D/2)

    pos_encode = torch.cat([encode_sin, encode_cos], dim=1)  # (M, D)
    return pos_encode


def get_great_circle_distance(
    lat_a: torch.Tensor, lon_a: torch.Tensor, lat_b: torch.Tensor, lon_b: torch.Tensor
) -> torch.Tensor:
    """Calculate the great-circle distance between two points on a sphere via the Haversine formula. Latitude and longitude values are used as inputs.

    Args:
        lat_a (torch.Tensor): Latitude of first point.
        lon_a (torch.Tensor): Longitude of first point.
        lat_b (torch.Tensor): Latitude of second point.
        lon_b (torch.Tensor): Longitude of second point.

    Returns:
        torch.Tensor: Tensor of great-circle distance between pairs of points multiplied by the radius of the earth.
    """
    delta_lat = torch.deg2rad(lat_a) - torch.deg2rad(lat_b)
    delta_lon = torch.deg2rad(lon_a) - torch.deg2rad(lon_b)
    # "Haversine" formula multiplied by the radius of the earth
    great_circle_dist = (
        2
        * 6371
        * torch.asin(
            torch.sqrt(
                torch.sin(delta_lat / 2) ** 2
                + torch.cos(torch.deg2rad(lat_a))
                * torch.cos(torch.deg2rad(lat_b))
                * torch.sin(delta_lon / 2) ** 2
            )
        )
    )
    return great_circle_dist


def get_2d_patched_lat_lon_from_grid(
    encode_dim: int, grid: torch.Tensor, patch: tuple
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculates 2D patched position encoding from grid. For each patch the mean latitute and longitude values are calculated.

    Args:
        encode_dim (int): Output encoding dimension `D`.
        grid (torch.Tensor): Latitude-longitude grid of dimensions `(2, 1, H, W)`
        patch (tuple): Patch dimensions. Different x- and y-values are supported.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Returns positional encoding tensor and scale tensor of shape `(H/patch[0] * W/patch[1], D)`.
    """
    # encode_dim has to be % 4 (lat-lon split, sin-cosine split)
    assert encode_dim % 4 == 0
    assert grid.dim() == 4

    # take the 2D pooled values of the mesh - this is the same as subsequent 1D pooling over x and y axis
    grid_h = F.avg_pool2d(grid[0], patch)
    grid_w = F.avg_pool2d(grid[1], patch)

    # get min and max values for x and y coordinates to calculate the diagonal of each patch
    grid_lat_max = F.max_pool2d(grid[0], patch)
    grid_lat_min = -F.max_pool2d(-grid[0], patch)
    grid_lon_max = F.max_pool2d(grid[1], patch)
    grid_lon_min = -F.max_pool2d(-grid[1], patch)
    grid_great_circle_dist = get_great_circle_distance(
        grid_lat_min, grid_lon_min, grid_lat_max, grid_lon_max
    )

    # use half of dimensions to encode grid_h
    encode_h = get_1d_pos_encode(encode_dim // 2, grid_h)  # (H*W/patch**2, D/2)
    encode_w = get_1d_pos_encode(encode_dim // 2, grid_w)  # (H*W/patch**2, D/2)
    # use all dimensions to encode scale
    scale_encode = get_1d_pos_encode(encode_dim, grid_great_circle_dist)  # (H*W/patch**2, D)

    pos_encode = torch.cat((encode_h, encode_w), axis=1)  # (H*W/patch**2, D)
    return pos_encode, scale_encode


def get_lat_lon_grid(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    """Return meshgrid of latitude and longitude coordinates. torch.meshgrid(*tensors, indexing='xy') the same behavior as calling numpy.meshgrid(*arrays, indexing=’ij’).
    lat = torch.tensor([1, 2, 3])
    lon = torch.tensor([4, 5, 6])
    grid_x, grid_y = torch.meshgrid(lat, lon, indexing='xy')
    grid_x = tensor([[1, 2, 3], [1, 2, ,3], [1, 2, 3]])
    grid_y = tensor([[4, 4, 4], [5, 5, ,5], [6, 6, 6]])
    Args:
        lat (torch.Tensor): 1D tensor of latitude values
        lon (torch.Tensor): 1D tensor of longitude values
    Returns:
        torch.Tensor: Meshgrid of shape `(2, 1, lat.shape, lon.shape)`
    """
    assert lat.dim() == 1
    assert lon.dim() == 1
    grid = torch.meshgrid(lat, lon, indexing="xy")
    grid = torch.stack(grid, axis=0)
    grid = grid.reshape(2, 1, lat.shape[-1], lon.shape[-1])

    return grid


def get_2d_patched_lat_lon_encode(
    encode_dim: int,
    lat: torch.Tensor,
    lon: torch.Tensor,
    patch: Union[int, list, tuple],
) -> torch.Tensor:
    """Positional encoding of latitude - longitude data.

    Args:
        encode_dim (int): Output encoding dimension `D`.
        lat (torch.Tensor): Tensor of latitude values `H`.
        patch (Union[list, tuple]): Patch dimensions. Different x- and y-values are supported.
        lon (torch.Tensor): Tensor of longitude values `W`.
        device (torch.cuda.device): Device.

    Returns:
        torch.Tensor: Returns positional encoding tensor of shape `(H/patch[0] * W/patch[1], D)`.
    """

    grid = get_lat_lon_grid(lat, lon)
    pos_encode, scale_encode = get_2d_patched_lat_lon_from_grid(encode_dim, grid, to_2tuple(patch))

    return pos_encode, scale_encode


###################################
# FROM HERE ON CODE IS DEPRECATED #
###################################

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_lat_lon_embed(embed_dim, lat, lon, device, cls_token=False):
    """
    Embed arbitrary latitude - longitude data.
    """
    grid = torch.meshgrid(lon.to(device), lat.to(device), indexing="xy")
    grid = torch.stack(grid, axis=0)
    grid = grid.reshape(2, 1, lat.shape[-1], lon.shape[-1])
    pos_embed = get_2d_sincos_pos_embed_from_grid_pytorch(embed_dim, grid)

    if cls_token:
        pos_embed = torch.cat((torch.zeros((1, embed_dim), device=device), pos_embed), axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_pytorch(embed_dim, grid_size_h, grid_size_w, device, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = torch.arange(grid_size_h, device=device)
    grid_w = torch.arange(grid_size_w, device=device)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, axis=0)

    grid = grid.reshape(2, 1, grid_size_h, grid_size_w)
    pos_embed = get_2d_sincos_pos_embed_from_grid_pytorch(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat((torch.zeros((1, embed_dim), device=device), pos_embed), axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid_pytorch(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_pytorch(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_pytorch(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat((emb_h, emb_w), axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_pytorch(embed_dim, pos, dtype=torch.float32):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=dtype, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    # out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
    out = torch.outer(pos, omega)

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# TODO(Cris): We have two functions with the same name in this folder.
#             We should remove one of them.
# def get_1d_sincos_pos_embed_from_grid_pytorch(embed_dim, pos, dtype=torch.float32):
#     """
#     embed_dim: output dimension for each position
#     pos: a list of positions to be encoded: size (M,)
#     out: (M, D)
#     """
#     assert embed_dim % 2 == 0
#     omega = torch.arange(embed_dim // 2, dtype=dtype).to(pos.device)
#     omega /= embed_dim / 2.0
#     omega = 1.0 / 10000**omega  # (D/2,)

#     pos = pos.reshape(-1)  # (M,)
#     # out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
#     out = torch.outer(pos, omega)

#     emb_sin = torch.sin(out)  # (M, D/2)
#     emb_cos = torch.cos(out)  # (M, D/2)

#     emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
#     return emb


def get_1d_sincos_pos_embed_from_grid_pytorch_stable(
    dim, timesteps, dtype=torch.float32, max_period=10000
):
    """
    Create sinusoidal timestep embeddings.
    Arguments:
        - `timesteps`: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
        - `dim`: the dimension of the output.
        - `max_period`: controls the minimum frequency of the embeddings.
    Returns:
        - embedding: [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=dtype) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model, new_size=(64, 128)):
    if "net.pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["net.pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        orig_num_patches = pos_embed_checkpoint.shape[-2]
        patch_size = model.patch_size
        w_h_ratio = new_size[1] // new_size[0]
        orig_h = int((orig_num_patches // w_h_ratio) ** 0.5)
        orig_w = w_h_ratio * orig_h
        orig_size = (orig_h, orig_w)
        new_size = (new_size[0] // patch_size, new_size[1] // patch_size)
        # print (orig_size)
        # print (new_size)
        if orig_size[0] != new_size[0]:
            print(
                "Interpolate PEs from %dx%d to %dx%d"
                % (orig_size[0], orig_size[1], new_size[0], new_size[1])
            )
            pos_tokens = pos_embed_checkpoint.reshape(
                -1, orig_size[0], orig_size[1], embedding_size
            ).permute(0, 3, 1, 2)
            new_pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size[0], new_size[1]), mode="bicubic", align_corners=False
            )
            new_pos_tokens = new_pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            checkpoint_model["net.pos_embed"] = new_pos_tokens
