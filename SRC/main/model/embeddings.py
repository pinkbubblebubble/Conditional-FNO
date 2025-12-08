import torch
import torch.nn as nn
from typing import Tuple, List


import math
import torch

class SinusoidalEmbedding(nn.Module):
    """
    SinusoidalEmbedding builds sin/cos positional features
    and caches the frequency vector for the MRU (most recently used) setup.

    Parameters
    ----------
    target_dim : int
        output embedding dim
    max_period : int
        controls lowest frequency ~ 1/max_period
    """
    def __init__(self, in_dim: int, out_dim: int, num_freqs: int = 32, max_period: float = 10.0):
        super().__init__()
        assert out_dim > 0
        self.num_freqs = num_freqs
        self.max_period = max_period
        self.raw_dim = in_dim * num_freqs * 2

        # cache state
        exponent = torch.arange(num_freqs, dtype=torch.float32)
        freqs = torch.exp(-math.log(max_period) * exponent / (num_freqs - 1))
        self.register_buffer('freqs', freqs)

        # linear layer
        self.mlp = nn.Sequential(
            nn.Linear(self.raw_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        condition : torch.Tensor
            shape [batch_size, in_dim]

        Returns
        -------
        torch.Tensor
            shape [batch_size, out_dim]
        """
        args = condition.unsqueeze(-1) * self.freqs.view(1, 1, -1)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        emb = emb.reshape(condition.shape[0], -1)
        out = self.mlp(emb)
        return out
    

class RegularGridEmbedding(nn.Module):
    """A positional embedding as a regular ND grid
    """
    def __init__(self, dim: int, grid_boundaries: List[List[int]] = None):
        """
        GridEmbedding applies a simple positional embedding as a regular ND grid

        Parameters
        ----------
        dim : int
            dimensions of positional encoding to apply
        grid_boundaries : list, optional
            coordinate boundaries of input grid along each dim, by default [[0, 1]] * dim
        """
        super().__init__()
        self.dim = dim
        self.grid_boundaries = [[0., 1.]] * dim if grid_boundaries is None else grid_boundaries
        assert self.dim == len(self.grid_boundaries), f"Error: expected grid_boundaries to be\
            an iterable of length {self.dim}, received {self.grid_boundaries}"
        
        # cache state
        self._grid: List[torch.Tensor] = None
        self._cache_key: Tuple[Tuple[int, ...], torch.device, torch.dtype] = None
    
    def regular_grid(self, resolutions: List[int], grid_boundaries: List[List[int]]):
        """
        regular_grid generates a tensor of coordinate points that describe a bounded regular grid.
        Creates a dim x res_d1 x ... x res_dn stack of positional encodings A, where
        A[:,c1,c2,...] = [[d1,d2,...dn]] at coordinate (c1,c2,...cn) on a (res_d1, ...res_dn) grid. 

        Parameters
        ----------
        resolutions : List[int]
            resolution of the output grid along each dimension
        grid_boundaries : List[List[int]]
            List of pairs [start, end] of the boundaries of the regular grid. 
            Must correspond 1-to-1 with resolutions.

        Returns
        -------
        grid: tuple(Tensor)
        list of tensors describing positional encoding 
        """
        assert len(resolutions) == len(grid_boundaries), "Error: inputs must have same number of dimensions"

        meshgrid_inputs = list()
        for res, (start, stop) in zip(resolutions, grid_boundaries):
            meshgrid_inputs.append(torch.linspace(start, stop, res + 1)[:-1])
        grid = torch.meshgrid(*meshgrid_inputs, indexing='ij')
        return grid

    def grid(self, spatial_dims: torch.Size, device: str, dtype: torch.dtype):
        """grid generates ND grid needed for pos encoding
        and caches the grid associated with MRU resolution

        Parameters
        ----------
        spatial_dims : torch.Size
             sizes of spatial resolution
        device : literal 'cpu' or 'cuda:*'
            where to load data
        dtype : str
            dtype to encode data

        Returns
        -------
        torch.tensor
            output grids to concatenate 
        """
        # handle case of multiple train resolutions
        key = (tuple(spatial_dims), device, dtype)
        if self._grid is None or self._cache_key != key:
            grids_by_dim = self.regular_grid(spatial_dims, grid_boundaries=self.grid_boundaries)
            # add batch, channel dims
            grids_by_dim = [x.to(device).to(dtype).unsqueeze(0).unsqueeze(0) for x in grids_by_dim]
            self._grid = grids_by_dim
            self._cache_key = key

        return self._grid

    def forward(self, data):
        """
        Params
        --------
        data: torch.Tensor
            assumes shape batch, channels, x_1, x_2, ...x_n
        """
        # assert data.ndim == self.dim + 2, (
        #     f"Expected data.ndim == {self.dim+2} (B,C,{'x,'*(self.dim-1)}x), got {data.shape}"
        # )
        batch_size = data.shape[0]
        grids = self.grid(spatial_dims=data.shape[2:], device=data.device, dtype=data.dtype)
        grids = [g.expand(batch_size, *g.shape[1:]) for g in grids]
        out =  torch.cat((data, *grids), dim=1)
        return out
    

