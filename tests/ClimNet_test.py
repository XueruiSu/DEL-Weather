from modules.climnet import ClimNet
import torch
from configs.Climax_train_modelparam import * # hyperparameters
from utils.utilities3 import count_params, LpLoss

net = ClimNet(
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

print("net.encoder", count_params(net.encoder))
print("net.backbone", count_params(net.backbone))
print("net.decoder", count_params(net.decoder))
# torch.save(net.state_dict(), "/blob/weathers2/xuerui/Dual-Weather/project/DEL-Weather/checkpoints/model_weights_init_2.pth") 



