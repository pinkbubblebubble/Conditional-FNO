from typing import Tuple
import torch
import torch.nn as nn

from model.cfno_block import *
from model.embeddings import RegularGridEmbedding, SinusoidalEmbedding
from model.layers import ChannelMLP


class ConditionalFNO(nn.Module):
    def __init__(
        self,
        num_modes: Tuple[int],
        in_channel: int,
        out_channel: int,
        hidden_channel: int,
        param_channel: int,
        num_layers: int,
        lifting_channel_ratio: int = 2,
        projection_channel_ratio: int = 2,
        condition_embed_dim: int = 128,
        act: nn.Module = nn.GELU()
    ):
        super().__init__()
        self.dim = len(num_modes)
        self.num_layers = num_layers
        self.in_channel = in_channel + self.dim
        self.out_channel = out_channel
        self.hidden_channel = hidden_channel

        # positional embedding
        self.positional_embedding = RegularGridEmbedding(dim=self.dim)

        # condition embedding
        self.conditional_embedding = SinusoidalEmbedding(
            in_dim=param_channel,
            out_dim=condition_embed_dim
        )

        # lifting and projection
        self.lifting = ChannelMLP(self.in_channel, self.hidden_channel, self.hidden_channel * lifting_channel_ratio, act=act)
        self.projection = ChannelMLP(self.hidden_channel, self.out_channel, self.hidden_channel * projection_channel_ratio, act=act)

        # fno blocks
        self.fno_blocks = ConditionalFNOBlocks(
            channel=self.hidden_channel,
            condition_channel=condition_embed_dim,
            mlp_channel=self.hidden_channel * 2,
            num_layers=num_layers,
            num_modes=num_modes,
            act=act
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
            x: [B, in_channels, *spatial]
            condition: [B, C_cond] tensor
        """
        # positional embedding
        x = self.positional_embedding(x)

        # condition embedding
        cond_embd = self.conditional_embedding(condition)  # [B, condition_embed_dim]

        # lifting
        x = self.lifting(x)

        # FNO blocks with condition
        for layer_idx in range(self.num_layers):
            x = self.fno_blocks(x, cond_embd, layer_idx)

        # projection
        x = self.projection(x)
        return x


def main():
    # --- 1. 配置测试参数 ---
    Batch_Size = 8
    In_Channels = 1   
    Out_Channels = 1  
    Height = 64
    Width = 64
    
    Param_Dim = 2     
    Modes = (12, 12)  # 64x64 分辨率下的标准配置
    Hidden_Channels = 32

    print(f"{'='*20} 配置 {'='*20}")
    print(f"输入尺寸: (B={Batch_Size}, C={In_Channels}, H={Height}, W={Width})")
    print(f"Modes: {Modes}")
    print(f"Hidden Channels: {Hidden_Channels}")
    print(f"Condition 参数维度: {Param_Dim}")

    # --- 2. 初始化你的 Conditional FNO (3层) ---
    print(f"\n{'='*20} 模型 1: Conditional FNO (3层) {'='*20}")
    model_cond = ConditionalFNO(
        num_modes=Modes,               
        in_channel=In_Channels,        
        out_channel=Out_Channels,      
        hidden_channel=Hidden_Channels,             
        param_channel=Param_Dim,       
        num_layers=3,                  # 你的配置：3层
        condition_embed_dim=256        # 保持之前的 Embedding 大小
    )

    cond_params = sum(p.numel() for p in model_cond.parameters() if p.requires_grad)
    print(f"ConditionalFNO (3 layers) 参数数量: {cond_params:,}")

    # --- 3. 初始化对比用的 Standard FNO (6层) ---
    print(f"\n{'='*20} 模型 2: Standard FNO (6层) {'='*20}")
    try:
        from neuralop.models import FNO
        
        # 为了公平对比，Modes 和 Hidden Channels 保持一致，但层数翻倍
        model_standard = FNO(
            n_modes=Modes,             # (12, 12)
            hidden_channels=Hidden_Channels, # 32
            in_channels=In_Channels+2,   # 1
            out_channels=Out_Channels, # 1
            n_layers=6                 # 对比目标：6层
        )
        
        std_params = sum(p.numel() for p in model_standard.parameters() if p.requires_grad)
        print(f"Standard FNO (6 layers)   参数数量: {std_params:,}")
        
        # 计算差异
        diff = std_params - cond_params
        ratio = cond_params / std_params
        print(f"\n--- 对比结果 ---")
        print(f"差异: 标准 6层 FNO 比 你的 3层条件 FNO 多了 {diff:,} 个参数")
        print(f"比例: 你的模型参数量仅为 6层 FNO 的 {ratio:.1%}")

    except ImportError:
        print("未找到 neuralop 库，跳过对比模型加载。")
        print("请运行: pip install neuraloperator")

    # --- 4. 前向传播测试 (只测你的模型) ---
    print(f"\n{'='*20} 运行测试 (Conditional FNO) {'='*20}")
    x = torch.randn(Batch_Size, In_Channels, Height, Width)
    cond = torch.randn(Batch_Size, Param_Dim)

    try:
        y = model_cond(x, condition=cond)
        print("运行成功！")
        print(f"输入 x: {x.shape}")
        print(f"条件 c: {cond.shape}")
        print(f"输出 y: {y.shape}")
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()