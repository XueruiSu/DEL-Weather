from modules.climnet import ClimNet_UQ
import torch
from configs.Climax_train_modelparam import * # hyperparameters
from utils.utilities3 import count_params, LpLoss

net = ClimNet_UQ(
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

# torch.save(net.state_dict(), "/blob/weathers2/xuerui/Dual-Weather/project/DEL-Weather/checkpoints/model_weights_init_2.pth") 
print("net.encoder", (net.encoder))
print("net.backbone", (net.backbone))
print("net.decoder", (net.decoder))

print("net.encoder", count_params(net.encoder))
print("net.backbone", count_params(net.backbone))

print("net.encoder_hat1", count_params(net.encoder_hat1))
print("net.encoder_hat2", count_params(net.encoder_hat2))
print("net.decoder", count_params(net.decoder))


print("net", count_params(net))
print("UQ net", count_params(net.decoder)+count_params(net.encoder_hat1)+count_params(net.encoder_hat2))

# print(net)


