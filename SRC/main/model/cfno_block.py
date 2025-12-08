import torch
from torch import nn
from typing import Tuple
from neuralop.layers.spectral_convolution import SpectralConv
    
from model.layers import ChannelMLP, ChannelLinear, SoftGating


class ConditionalFNOBlocks(nn.Module):
    """
    """
    def __init__(
        self,
        channel: int,
        num_layers: int,
        num_modes: Tuple[int, ...],
        condition_channel: int,
        mlp_channel: int = None,
        act: nn.Module = nn.GELU(),
    ):
        super().__init__()
        self.channel = channel
        self.num_layers = num_layers
        self.num_modes = num_modes
        self.condition_channel = condition_channel
        self.mlp_channel = mlp_channel if mlp_channel is not None else channel * 2
        self.act = act

        # conditional projections
        self.condition_projs = nn.ModuleList([
            nn.Linear(self.condition_channel, self.channel * 2)
            for _ in range(num_layers)
        ])

        for proj in self.condition_projs:
            nn.init.constant_(proj.weight, 0)
            nn.init.constant_(proj.bias, 0)

        # skip connections for FNO branch and MLP branch
        self.spectral_skips = nn.ModuleList([ChannelLinear(self.channel, self.channel) for _ in range(num_layers)])
        self.local_skips = nn.ModuleList([SoftGating(self.channel) for _ in range(num_layers)])

        # spectral convs
        self.convs_pre = nn.ModuleList(
            [SpectralConv(self.channel, self.channel, n_modes=self.num_modes) for _ in range(num_layers)]
        )

        self.convs_post = nn.ModuleList(
            [SpectralConv(self.channel, self.channel, n_modes=self.num_modes) for _ in range(num_layers)]
        )

        # local MLPs
        self.channel_mlps = nn.ModuleList(
            [ChannelMLP(self.channel, self.channel, self.mlp_channel, act=self.act) for _ in range(num_layers)]
        )

    @staticmethod
    def _broadcast(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        while x.ndim < like.ndim:
            x = x.unsqueeze(-1)
        return x

    def forward(self, x: torch.Tensor, condition: torch.Tensor, index: int = 0) -> torch.Tensor:
        # skip connections for spectral branch and local branch
        x_skip_spectral = self.spectral_skips[index](x)
        x_skip_local = self.local_skips[index](x)

        # spectral conv
        x_spectral = self.convs_pre[index](x) # [B, C, *spatial]
        cond = self.condition_projs[index](condition) # [B, C]
        cond = self._broadcast(cond, x_spectral) # [B, C, *spatial]
        scale, shift = cond.chunk(2, dim=1)
        x_spectral = x_spectral * (1 + scale) + shift
        x_spectral = self.convs_post[index](x_spectral)
        x = x_spectral + x_skip_spectral
        if index < (self.num_layers - 1):
            x = self.act(x)

        # local MLP
        x = self.channel_mlps[index](x) + x_skip_local
        if index < (self.num_layers - 1):
            x = self.act(x)

        return x