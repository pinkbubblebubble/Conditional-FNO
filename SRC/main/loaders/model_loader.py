import torch
import torch.nn as nn
from typing import Tuple

class ConcatFNOWrapper(nn.Module):
    def __init__(self, fno_model: nn.Module):
        super().__init__()
        self.fno = fno_model

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        num_params = condition.shape[1]
        condition_reshaped = condition.view(B, num_params, 1, 1)
        condition_expanded = condition_reshaped.expand(B, num_params, H, W)
        x_cat = torch.cat([x, condition_expanded], dim=1)
        return self.fno(x_cat)

class NoConditionalFNOWrapper(nn.Module):
    def __init__(self, fno_model: nn.Module):
        super().__init__()
        self.fno = fno_model

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.fno(x)

def load_model(model_name: str, 
               num_modes: Tuple[int, ...],
               in_channel: int,
               out_channel: int,
               param_channel: int,
               num_layers: int, 
               hidden_channel: int = 32, 
               condition_embed_dim: int=128):
    """
    Load model by name.
    """
    if model_name == "CFNO":
        from model.conditional_fno import ConditionalFNO
        model = ConditionalFNO(
            num_modes=num_modes,
            in_channel=in_channel,
            out_channel=out_channel,
            hidden_channel=hidden_channel,
            num_layers=num_layers,
            param_channel=param_channel,
            condition_embed_dim=condition_embed_dim
        )

    elif model_name == "DFNO":
        from model.deepONet_fno import DeepONetStyleFNO
        model = DeepONetStyleFNO(
            num_modes=num_modes,
            in_channel=in_channel,
            out_channel=out_channel,
            hidden_channel=hidden_channel,
            num_layers=num_layers,
            param_channel=param_channel,
            condition_embed_dim=condition_embed_dim
        )
        
    elif model_name == "FNO":
        from neuralop.models import FNO
        raw_fno = FNO(
            n_modes=num_modes,
            hidden_channels=hidden_channel,
            in_channels=in_channel + param_channel, 
            out_channels=out_channel,
            n_layers=num_layers
        )
        model = ConcatFNOWrapper(raw_fno)
    elif model_name == "noFNO":
        from neuralop.models import FNO
        raw_fno = FNO(
            n_modes=num_modes,
            hidden_channels=hidden_channel,
            in_channels=in_channel, 
            out_channels=out_channel,
            n_layers=num_layers
        )
        model = NoConditionalFNOWrapper(raw_fno)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model