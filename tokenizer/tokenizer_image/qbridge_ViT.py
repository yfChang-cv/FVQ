# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


#################################################################################
#                                 Core ViT Model                                #
#################################################################################

class ViTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of VQBridge.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class VQBridge(nn.Module):
    """
    VQBridge model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=128,
        patch_size=4,
        in_channels=16,
        head_hidden_size=16,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        
        hidden_size = head_hidden_size * num_heads
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)

        self.blocks = nn.ModuleList([
            ViTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)


        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x):
        """
        Forward pass of VQBridge.
        x: (N, C, H, W) codebook
        """
        x = self.x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2
        for block in self.blocks:
            x = block(x)                      # (N, T, D)
        x = self.final_layer(x)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x


class QBridge_lin(nn.Module):
    def __init__(self, in_channels=256, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_channels, in_channels)
    def forward(self, x, y=None):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.linear(x)
        x = x.permute(0, 3, 1, 2)
        return x
    
class QBridge_MLP_5(nn.Module):
    def __init__(self, in_channels=256, **kwargs):
        super().__init__()
        self.linear_5 = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )
    def forward(self, x, y=None):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.linear_5(x)
        x = x.permute(0, 3, 1, 2)
        return x
    
class QBridge_none(nn.Module):
    def __init__(self, in_channels=256, **kwargs):
        super().__init__()
        
    def forward(self, x, y=None):
        return x

#################################################################################
#                                   VQBridge Configs                                  #
#################################################################################

def QBridge_XS_2(**kwargs):
    return VQBridge(depth=2, patch_size=2, head_hidden_size=8, num_heads=4, **kwargs)

def QBridge_S_2(**kwargs):
    return VQBridge(depth=2, patch_size=2, head_hidden_size=16, num_heads=4, **kwargs)

def QBridge_S_4(**kwargs):
    return VQBridge(depth=2, patch_size=4, head_hidden_size=16, num_heads=4, **kwargs)

def QBridge_S_4_d4(**kwargs):
    return VQBridge(depth=4, patch_size=4, head_hidden_size=16, num_heads=4, **kwargs)

def QBridge_S_8(**kwargs):
    return VQBridge(depth=2, patch_size=8, head_hidden_size=16, num_heads=4, **kwargs)

def QBridge_B_2(**kwargs):
    return VQBridge(depth=2, patch_size=2, head_hidden_size=32, num_heads=4, **kwargs)

def QBridge_B_8(**kwargs):
    return VQBridge(depth=2, patch_size=8, head_hidden_size=32, num_heads=4, **kwargs)

def QBridge_B_4(**kwargs):
    return VQBridge(depth=2, patch_size=4, head_hidden_size=32, num_heads=4, **kwargs)

def QBridge_B_4_d1(**kwargs):
    return VQBridge(depth=1, patch_size=4, head_hidden_size=32, num_heads=4, **kwargs)

def QBridge_B_4_d4(**kwargs):
    return VQBridge(depth=4, patch_size=4, head_hidden_size=32, num_heads=4, **kwargs)

def QBridge_L_2_d4(**kwargs):
    return VQBridge(depth=4, patch_size=2, head_hidden_size=64, num_heads=4, **kwargs)

def QBridge_L_8(**kwargs):
    return VQBridge(depth=2, patch_size=8, head_hidden_size=64, num_heads=4, **kwargs)

def QBridge_L_4(**kwargs):
    return VQBridge(depth=2, patch_size=4, head_hidden_size=64, num_heads=4, **kwargs)

def QBridge_L_2(**kwargs):
    return VQBridge(depth=2, patch_size=2, head_hidden_size=64, num_heads=4, **kwargs)

def QBridge_L_4_d1(**kwargs):
    return VQBridge(depth=1, patch_size=4, head_hidden_size=64, num_heads=4, **kwargs)

def QBridge_L_4_d4(**kwargs):
    return VQBridge(depth=4, patch_size=4, head_hidden_size=64, num_heads=4, **kwargs)

def QBridge_XL_4(**kwargs):
    return VQBridge(depth=2, patch_size=4, head_hidden_size=128, num_heads=4, **kwargs)

def QBridge_XL_4_d4(**kwargs):
    return VQBridge(depth=4, patch_size=4, head_hidden_size=128, num_heads=4, **kwargs)

def QBridge_lin_1(in_channels=256, **kwargs):
    return QBridge_lin(in_channels)

def QBridge_lin_5(in_channels=256, **kwargs):
    return QBridge_MLP_5(in_channels)

def QBridge_XL_8(**kwargs):
    return VQBridge(depth=2, patch_size=8, head_hidden_size=128, num_heads=4, **kwargs)


def QBridge_siglip2_B_16_256(**kwargs):
    return VQBridge(depth=2, patch_size=4, head_hidden_size=192, num_heads=4, **kwargs)

def QBridge_d4_siglip2_B_16_256(**kwargs):
    return VQBridge(depth=4, patch_size=4, head_hidden_size=192, num_heads=4, **kwargs)

QBridge_models = {
    'Qbridge-none': QBridge_none,
    'Qbridge-lin/1': QBridge_lin_1,
    'Qbridge-lin/5': QBridge_lin_5,
    'QBridge-XS/2': QBridge_XS_2,
    'QBridge-S/2': QBridge_S_2,
    'QBridge-S/4': QBridge_S_4,
    'QBridge-S/8': QBridge_S_8,
    'QBridge-B/2': QBridge_B_2,
    'QBridge-B/8': QBridge_B_8,
    'QBridge-B/4': QBridge_B_4,
    'QBridge-B/4-d1': QBridge_B_4_d1,
    'QBridge-B/4-d4': QBridge_B_4_d4,
    'Qbridge-S/4-d4': QBridge_S_4_d4,
    'QBridge-L/4-d4': QBridge_L_4_d4,
    'QBridge-L/2-d4': QBridge_L_2_d4,
    'QBridge-L/8': QBridge_L_8,
    'QBridge-L/4': QBridge_L_4,
    'QBridge-L/2': QBridge_L_2,
    'QBridge-L/4-d1': QBridge_L_4_d1,
    'QBridge-XL/4': QBridge_XL_4,
    'QBridge-XL/4-d4': QBridge_XL_4_d4,
    
    'QBridge-siglip2-B-p16-256/4': QBridge_siglip2_B_16_256,
    'QBridge-d4-siglip2-B-p16-256/4': QBridge_d4_siglip2_B_16_256, 
    'QBridge-XL/8': QBridge_XL_8,
}


if __name__ == '__main__':
    import timm.models.vision_transformer as vit

    # 临时禁用 fused attention
    original_use_fused_attn = vit.use_fused_attn
    vit.use_fused_attn = lambda: False
    def format_flops(flops):
        """将FLOP数值转换为人类可读格式"""
        if flops >= 1e12:
            return f"{flops / 1e12:.2f}T"
        elif flops >= 1e9:
            return f"{flops / 1e9:.2f}G"
        elif flops >= 1e6:
            return f"{flops / 1e6:.2f}M"
        elif flops >= 1e3:
            return f"{flops / 1e3:.2f}K"
        else:
            return f"{flops:.0f}"

    name = 'QBridge-L/4'
    # name = 'QBridge-L/8'
    # name = "Qbridge-lin/5"
    # codebooksize = 262144
    codebooksize=16384
    channel = 256
    h_ = int(codebooksize**0.5)
    input = (torch.rand(1, channel, h_, h_), )
    model = QBridge_models[name](input_size=h_, in_channels=channel)



    from fvcore.nn import FlopCountAnalysis
    model.eval()
    
    flops = FlopCountAnalysis(model, input)
    total_flops = flops.total()
    readable_flops = format_flops(total_flops)
    print(f"Total FLOPs: {readable_flops}")
    print(f"Total FLOPs (raw): {total_flops:,}")
    
    
    # 获取每个模块的详细统计
    flop_counts = flops.by_module()
    
    print("=== 各模块FLOP统计 ===")
    for module_name, flop_count in flop_counts.items():
        print(f"{module_name}: {flop_count:,}")
    
    # 获取每个算子的统计
    flop_counts_ops = flops.by_operator()
    print("\n=== 各算子FLOP统计 ===")
    for op_name, flop_count in flop_counts_ops.items():
        print(f"{op_name}: {flop_count:,}")
    
    # 获取不支持的算子
    unsupported = flops.unsupported_ops()
    print(f"\n=== 不支持的算子 ===")
    for op_name, count in unsupported.items():
        print(f"{op_name}: 出现 {count} 次")
    
    # from ptflops import get_model_complexity_info
    
    # # 设置模型为评估模式
    # model.eval()
    
    # # 计算FLOP和参数量
    # macs, params = get_model_complexity_info(
    #     model, 
    #     ((16, 128, 128), ),  # 输入形状 (C, H, W)
    #     print_per_layer_stat=True,
    #     as_strings=True
    # )
    
    # print(f'Computational complexity: {macs}')
    # print(f'Number of parameters: {params}')


