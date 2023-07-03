import pytest
import torch
from configs.Climax_train_modelparam import * # hyperparameters
from data import Metadata
from utils.metrics import lat_weighted_mse
from modules.climnet import ClimNet

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
    """Test forward pass of the ClimNet module with cmip6 variables and pressure levels."""
    bs = 2
    model = ClimNet(
        const_vars=tuple(const_vars),
        single_vars=tuple(single_vars),
        atmos_vars=tuple(atmos_vars),
        atmos_levels=tuple(atmos_levels),
        patch_size=16,
        embed_dim=768,
        depth=12,
        decoder_depth=2,
        num_heads=12,
        mlp_ratio=4,
        drop_path=0.1,
        drop_rate=0.1,
        use_flash_attn=True,
    )
    single_inputs = torch.randn(bs, 1, len(single_vars), *grid_size)
    single_outputs = torch.randn(bs, len(single_vars), *grid_size)

    atmos_inputs = {}
    atmos_outputs = {}
    for var in atmos_vars:
        atmos_inputs[var] = torch.randn(bs, 1, len(atmos_vars), *grid_size)
        atmos_outputs[var] = torch.randn(bs, len(atmos_vars), *grid_size)

    lead_times = torch.randint(1, 7, (bs,)).to(dtype=single_inputs.dtype)
    metadata = Metadata(
        tuple(single_vars),
        {k: atmos_levels for k in atmos_levels},
        torch.rand(grid_size[0]).sort()[0] * 180,
        torch.rand(grid_size[1]).sort()[0] * 360,
    )

    _, single_out, atmos_out = model(single_inputs, atmos_inputs, lead_times, metadata)
    assert single_out.shape == single_outputs.shape

    assert atmos_outputs.keys() == atmos_out.keys()
    for var in atmos_outputs.keys():
        assert atmos_outputs[var].shape == atmos_out[var].shape

    loss_dict, single_out, atmos_out = model(
        single_inputs,
        atmos_inputs,
        lead_times,
        metadata,
        single_outputs,
        atmos_outputs,
        metric=[lat_weighted_mse],
    )
    
    
    assert len(loss_dict) == 1
    
    

