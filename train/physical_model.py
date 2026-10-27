from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..neural_networks import ParameterKAN, ParameterLSTM
from ..phy_models import CryoHydro


class CryoHydroModel(nn.Module):
    """CryoHydro: dynamic LSTM and separate static/routing KANs."""

    def __init__(self, config: dict[str, Any], device: torch.device | str | None = None) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(device or config.get("device", "cpu"))
        phy_cfg = config.get("phy_model", config)
        nn_cfg = config.get("nn_model", {})
        self.phy_model = CryoHydro(phy_cfg, device=self.device)
        input_size = int(nn_cfg.get("input_size", len(nn_cfg.get("forcings", [])) + len(nn_cfg.get("attributes", []))))
        static_input_size = int(nn_cfg.get("static_input_size", len(phy_cfg.get("attributes", []))))
        routing_input_size = int(nn_cfg.get("routing_input_size", static_input_size))
        self.param_chunk_size = int(nn_cfg.get("param_chunk_size", 0))
        self.nmul = int(nn_cfg.get("nmul", phy_cfg.get("nmul", 1)))
        self.dynamic_params = list(nn_cfg.get("dynamic_params", self.phy_model.dynamic_params))
        self.static_params = list(nn_cfg.get("static_params", []))
        self.routing_params = list(nn_cfg.get("routing_params", []))
        self.nn_model = ParameterLSTM(
            input_size=input_size,
            output_size=len(self.dynamic_params) * self.nmul,
            hidden_size=int(nn_cfg.get("hidden_size", 128)),
            num_layers=int(nn_cfg.get("num_layers", 1)),
            dropout=float(nn_cfg.get("dropout", 0.1)),
        ).to(self.device)
        self.static_model = ParameterKAN(
            input_size=static_input_size,
            output_size=len(self.static_params) * self.nmul,
            hidden_size=int(nn_cfg.get("static_hidden_size", 1024)),
            dropout=float(nn_cfg.get("static_dropout", 0.1)),
            grid_size=int(nn_cfg.get("static_grid_size", nn_cfg.get("kan_grid_size", 8))),
        ).to(self.device)
        self.routing_model = ParameterKAN(
            input_size=routing_input_size,
            output_size=len(self.routing_params),
            hidden_size=int(nn_cfg.get("routing_hidden_size", 64)),
            dropout=float(nn_cfg.get("routing_dropout", 0.1)),
            grid_size=int(nn_cfg.get("routing_grid_size", nn_cfg.get("kan_grid_size", 8))),
        ).to(self.device)

    def generate_dynamic_parameters(self, nn_input: torch.Tensor, state=None, return_state: bool = False):
        nn_input = nn_input.to(self.device)
        if self.param_chunk_size > 0 and nn_input.shape[1] > self.param_chunk_size:
            chunks, hidden, cell = [], [], []
            for start in range(0, nn_input.shape[1], self.param_chunk_size):
                end = min(start + self.param_chunk_size, nn_input.shape[1])
                local_state = None if state is None else tuple(v[:, start:end].contiguous() for v in state)
                output, next_state = self.nn_model(nn_input[:, start:end], local_state, return_state=True)
                chunks.append(output)
                hidden.append(next_state[0])
                cell.append(next_state[1])
            output = torch.cat(chunks, dim=1)
            next_state = (torch.cat(hidden, dim=1), torch.cat(cell, dim=1))
            return (output, next_state) if return_state else output
        return self.nn_model(nn_input, state, return_state=return_state)

    def generate_static_parameters(
        self,
        static_input: torch.Tensor,
        routing_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_static = self.static_model(static_input.to(self.device))
        raw_routing = self.routing_model(routing_input.to(self.device))
        return raw_static, raw_routing

    def generate_parameters(
        self,
        nn_input: torch.Tensor,
        static_input: torch.Tensor,
        routing_input: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        raw_dynamic = self.generate_dynamic_parameters(nn_input)
        raw_static, raw_routing = self.generate_static_parameters(static_input, routing_input)
        return {
            "dynamic": raw_dynamic,
            "static": raw_static,
            "routing": raw_routing,
        }

    def forward(self, data_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        nn_input = data_dict.get("xc_nn_norm", data_dict.get("x_nn", data_dict["x_phy"]))
        static_input = data_dict["c_norm"]
        routing_input = data_dict["r_norm"]
        parameters = self.generate_parameters(nn_input, static_input, routing_input)
        return self.phy_model(data_dict, parameters)
