import pytest
import torch
from configs.Climax_train_modelparam import * # hyperparameters
from data import Metadata
from utils.metrics import lat_weighted_mse
from modules.climnet import ClimNet_UQ
import torch.nn.functional as F

# TODO(Johannes): put tests on GPU
@pytest.mark.parametrize(
    "grid_size",
    [
        (32, 64),
        (128, 256),
        (320, 720),
    ],
)
@pytest.mark.skip(reason="Forward pass seems to fail when doing CI.")
def test_climnet_forward(grid_size):
    """Test forward pass of the ClimNet module with era5 variables and pressure levels."""
    bs = 2
    model = ClimNet_UQ(
        const_vars=tuple(const_vars),
        single_vars=tuple(single_vars),
        atmos_vars=tuple(atmos_vars),
        atmos_levels=tuple(atmos_levels),
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        decoder_depth=decoder_depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path=drop_path,
        drop_rate=drop_rate,
        use_flash_attn=use_flash_attn,
        
        embed_dim_UQ=embed_dim_UQ,
        out_embed_dim_UQ=out_embed_dim_UQ,
        depth_UQ=depth_UQ,
        en_UQ_embed_dim=en_UQ_embed_dim,
        num_heads_UQ_en=num_heads_UQ_en,
        num_heads_UQ_down=num_heads_UQ_down,
        mlp_ratio_UQ=mlp_ratio_UQ,
        drop_path_UQ=drop_path_UQ,
        drop_rate_UQ=drop_rate_UQ,
        use_flash_attn_UQ=use_flash_attn_UQ,
        decoder_depth_down_UQ=decoder_depth_down_UQ,
    )
    single_inputs = torch.randn(bs, 1, len(single_vars), *grid_size)
    single_outputs = torch.randn(bs, len(single_vars), *grid_size)

    atmos_inputs = {}
    atmos_outputs = {}
    for var in atmos_vars:
        atmos_inputs[var] = torch.randn(bs, 1, len(atmos_levels), *grid_size)
        atmos_outputs[var] = torch.randn(bs, len(atmos_levels), *grid_size)

    lead_times = torch.randint(1, 7, (bs,)).to(dtype=single_inputs.dtype)
    metadata = Metadata(
        tuple(single_vars),
        {k: atmos_levels for k in atmos_vars},
        torch.rand(grid_size[0]).sort()[0] * 720,
        torch.rand(grid_size[1]).sort()[0] * 1440,
    )

    R2Loss_target = torch.randn(bs, len(atmos_vars)*len(atmos_levels)+len(single_vars))
    
    _, R2Loss_pre = model(single_inputs, atmos_inputs, lead_times, metadata, single_outputs, atmos_outputs)
    assert R2Loss_pre.shape == R2Loss_target.shape

    loss_dict, R2Loss_pre = model(
        single_inputs,
        atmos_inputs,
        lead_times,
        metadata,
        single_outputs,
        atmos_outputs,
        R2Loss_target,
        metric=[F.mse_loss],
    )
    print("single_inputs", single_inputs.shape)
    for k in atmos_inputs:
        print("atmos_inputs", k, atmos_inputs[k].shape)
    print("single_outputs", single_outputs.shape)
    for k in atmos_outputs:
        print("atmos_outputs", k, atmos_outputs[k].shape)
    print("lead_times", lead_times.shape)
    print("metadata lat", metadata.lat.shape)
    print("metadata lon", metadata.lon.shape)
    print("loss_dict", loss_dict)
    assert len(loss_dict) == 1
    
    
if __name__ == "__main__":
    test_climnet_forward(grid_size=(720, 1440))
