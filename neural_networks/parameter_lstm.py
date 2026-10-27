from __future__ import annotations

import torch
from torch import nn


class ParameterLSTM(nn.Module):
    """LSTM parameter generator for differentiable process models.

    Input shape is [time, batch, features]. Output shape is
    [time, batch, n_parameters]. The physical model is responsible for
    applying sigmoid and mapping parameters to physical ranges.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor, state=None, return_state: bool = False):
        # State is explicit: unrelated training windows never share hidden state.
        z, next_state = self.lstm(x, state)
        output = self.head(z)
        return (output, next_state) if return_state else output
