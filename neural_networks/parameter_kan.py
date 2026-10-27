from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class KANLayer(nn.Module):
    """Lightweight KAN-style layer with learnable edge functions.

    Each input feature is expanded on fixed radial basis functions, then each
    output receives a learned weighted sum over input-feature bases. This keeps
    the same tensor convention as Linear: [..., input_size] -> [..., output_size].
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        grid_size: int = 8,
        grid_min: float = -3.0,
        grid_max: float = 3.0,
        base_scale: float = 0.1,
        spline_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.grid_size = grid_size

        grid = torch.linspace(grid_min, grid_max, grid_size)
        self.register_buffer("grid", grid)
        self.bandwidth = float(grid_max - grid_min) / max(grid_size - 1, 1)

        self.base_weight = nn.Parameter(torch.empty(output_size, input_size))
        self.spline_weight = nn.Parameter(torch.empty(output_size, input_size, grid_size))
        self.bias = nn.Parameter(torch.zeros(output_size))

        nn.init.xavier_uniform_(self.base_weight, gain=base_scale)
        nn.init.normal_(self.spline_weight, mean=0.0, std=spline_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.input_size)

        base = F.silu(x_flat) @ self.base_weight.t()
        basis = torch.exp(-((x_flat.unsqueeze(-1) - self.grid) / (self.bandwidth + 1e-6)).square())
        spline = torch.einsum("nig,oig->no", basis, self.spline_weight)

        y = base + spline + self.bias
        return y.reshape(*original_shape, self.output_size)


class ParameterKAN(nn.Module):
    """KAN parameter generator for basin-fixed physical parameters.

    Input shape is [batch, features]. Output shape is [batch, n_parameters].
    The physical model maps raw outputs to physical parameter ranges.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 64,
        dropout: float = 0.0,
        grid_size: int = 8,
    ) -> None:
        super().__init__()
        self.layer1 = KANLayer(input_size, hidden_size, grid_size=grid_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = KANLayer(hidden_size, output_size, grid_size=grid_size, base_scale=0.01, spline_scale=0.005)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.layer1(x)
        z = torch.tanh(self.norm(z))
        z = self.dropout(z)
        return self.layer2(z)
