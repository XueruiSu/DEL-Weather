import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.helpers import to_2tuple
from timm.models.vision_transformer import trunc_normal_
from einops import rearrange


def get_conv2d_weights(
    in_channels,
    out_channels,
    kernel_size,
):
    weight = torch.empty(out_channels, in_channels, *kernel_size)
    return weight


def get_conv2d_biases(out_channels):
    bias = torch.empty(out_channels)
    return bias


class GroupedVarPatchEmbed(nn.Module):
    def __init__(
        self, max_vars: int, img_size, patch_size, embed_dim, norm_layer=None, flatten=True
    ):
        super().__init__()
        self.max_vars = max_vars
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.flatten = flatten

        weights = torch.stack(
            [get_conv2d_weights(1, embed_dim, self.patch_size) for _ in range(max_vars)], dim=0
        )
        self.proj_weights = nn.Parameter(weights)
        biases = torch.stack([get_conv2d_biases(embed_dim) for _ in range(max_vars)], dim=0)
        self.proj_biases = nn.Parameter(biases)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for idx in range(self.max_vars):
            nn.init.kaiming_uniform_(self.proj_weights[idx], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.proj_weights[idx])
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.proj_biases[idx], -bound, bound)

    def forward(self, x, vars=None):
        B, C, H, W = x.shape
        weights = self.proj_weights[vars].flatten(0, 1)
        biases = self.proj_biases[vars].flatten(0, 1)

        groups = self.max_vars if vars is None else len(vars)
        proj = F.conv2d(x, weights, biases, groups=groups, stride=self.patch_size)
        if self.flatten:
            proj = proj.reshape(B, groups, -1, *proj.shape[-2:])
            proj = proj.flatten(3).transpose(2, 3)

        x = self.norm(proj)
        return x


class FlexGroupedVarPatchEmbed(nn.Module):
    def __init__(self, max_vars: int, patch_size, embed_dim, norm_layer=None, flatten=True):
        super().__init__()
        self.max_vars = max_vars
        self.patch_size = to_2tuple(patch_size)
        self.flatten = flatten

        weights = torch.stack(
            [get_conv2d_weights(1, embed_dim, self.patch_size) for _ in range(max_vars)], dim=0
        )
        self.proj_weights = nn.Parameter(weights)
        biases = torch.stack([get_conv2d_biases(embed_dim) for _ in range(max_vars)], dim=0)
        self.proj_biases = nn.Parameter(biases)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for idx in range(self.max_vars):
            nn.init.kaiming_uniform_(self.proj_weights[idx], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.proj_weights[idx])
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.proj_biases[idx], -bound, bound)

    def forward(self, x, vars=None):
        B, *_ = x.shape
        weights = self.proj_weights[vars].flatten(0, 1)
        biases = self.proj_biases[vars].flatten(0, 1)

        groups = self.max_vars if vars is None else len(vars)
        # TODO(Cris): When using fp16, this returns a tensor with NaNs. Why?
        proj = F.conv2d(x, weights, biases, groups=groups, stride=self.patch_size)
        if self.flatten:
            proj = proj.reshape(B, groups, -1, *proj.shape[-2:])
            proj = proj.flatten(3).transpose(2, 3)

        x = self.norm(proj)
        return x


class StableGroupedVarPatchEmbed(nn.Module):
    def __init__(
        self,
        max_vars: int,
        patch_size: int,
        embed_dim: int,
        norm_layer: nn.Module = None,
        return_flatten: bool = True,
    ):
        super().__init__()
        self.max_vars = max_vars
        self.patch_size = to_2tuple(patch_size)
        self.embed_dim = embed_dim
        self.return_flatten = return_flatten

        self.proj = nn.ModuleList(
            [
                nn.Conv2d(
                    1,
                    embed_dim,
                    kernel_size=patch_size,
                    stride=patch_size,
                    bias=True if norm_layer is None else False,
                )
                for _ in range(max_vars)
            ]
        )

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

        self.apply(self.initialize)

    def initialize(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, vars: Iterable[int]):
        """Forward fucntion

        Args:
            x (torch.Tensor): a shape of [BT, V, L, C] tensor
            vars (list[int], optional): a list of variable ID

        Returns:
            proj (torch.Tensor): a shape of [BT V L' C] tensor
        """
        proj = []
        for i, var in enumerate(vars):
            proj.append(self.proj[var](x[:, i : i + 1]))
        proj = torch.stack(proj, dim=1)  # BT, V, C, H, W

        if self.return_flatten:
            proj = rearrange(proj, "b v c h w -> b v (h w) c")

        proj = self.norm(proj)

        return proj
