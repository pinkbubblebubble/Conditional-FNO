import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP).

    A simple fully-connected network with configurable depth, activation,
    and optional dropout between hidden layers.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension.
    hidden_dim : int, optional
        Hidden layer dimension. Default = out_dim.
    num_layers : int, optional
        Number of linear layers. Default = 2.
    act : nn.Module, optional
        Activation function. Default = nn.SiLU().
    dropout : float, optional
        Dropout probability between hidden layers. Default = 0.0.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = None,
        num_layers: int = 2,
        act: nn.Module = nn.SiLU(),
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_layers >= 1
        if hidden_dim is None:
            hidden_dim = out_dim

        layers = []
        last_dim = in_dim
        for i in range(num_layers):
            next_dim = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(last_dim, next_dim))
            if i < num_layers - 1:
                layers.append(act)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            last_dim = next_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [N, in_dim].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [N, out_dim].
        """
        return self.net(x)
    

class ChannelLinear(nn.Module):
    """
    Channel-wise linear layer.

    Applies a 1x1 convolution (pointwise linear mapping) over the channel
    dimension while keeping spatial dimensions unchanged.

    Parameters
    ----------
    in_channel : int
        Number of input channels.
    out_channel : int
        Number of output channels.
    """
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.proj = nn.Conv1d(in_channel, out_channel, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, in_channel, *spatial].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, out_channel, *spatial].
        """
        batch_size, in_channel = x.shape[:2]
        spatial_dims = x.shape[2:]
        x = x.reshape(batch_size, in_channel, -1) # (batch_size, in_channel, S)
        y = self.proj(x) # (batch_size, out_channel, S)
        y = y.reshape(batch_size, y.shape[1], *spatial_dims)  # (batch_size, out_channel, *spatial_dims)
        return y


class ChannelMLP(nn.Module):
    """
    Channel-wise multi-layer perceptron (MLP).

    A simple channel mixing network applied independently to each spatial
    location. It uses ChannelLinear layers with configurable depth and
    activation.

    Parameters
    ----------
    in_channel : int
        Number of input channels.
    out_channel : int
        Number of output channels.
    hidden_channel : int
        Hidden channel dimension.
    num_layers : int, optional
        Number of ChannelLinear layers. Default = 2.
    act : nn.Module, optional
        Activation function. Default = nn.GELU().
    """
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        hidden_channel: int,
        num_layers: int = 2,
        act: nn.Module = nn.GELU(),
    ):
        super().__init__()
        assert num_layers >= 2, "num_layers must be >= 2"

        layers = []
        in_c = in_channel
        for i in range(num_layers):
            out_c = out_channel if i == num_layers - 1 else hidden_channel
            layers.append(ChannelLinear(in_c, out_c))
            if i < num_layers - 1:
                layers.append(act)
            in_c = out_c

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [N, C_in, *spatial].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [N, C_out, *spatial].
        """
        return self.net(x)
    
class SoftGating(nn.Module):
    """
    Soft gating layer.

    Applies a learnable, channel-wise scaling (and optional bias) to the input
    tensor. This layer performs an element-wise affine transformation along the
    channel dimension and is often used for adaptive feature reweighting.

    Parameters
    ----------
    channels : int
        Number of input channels.
    bias : bool, optional
        If True, adds a learnable bias term. Default = False.
    """
    def __init__(self, channels: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [N, C, *spatial].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [N, C, *spatial] after channel-wise scaling.
        """
        shape = (1, -1) + (1,) * (x.ndim - 2)
        y = x * self.weight.view(*shape)
        if self.bias is not None:
            y = y + self.bias.view(*shape)
        return y