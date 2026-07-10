# Copyright 2026
"""BEFT prior-head modules for vLLM-BERAG inference."""

from __future__ import annotations

import torch
import torch.nn as nn


class BeftPriorHead(nn.Module):
    """Prior head matching LLaMA-Factory BEFT `prior_head.pt` checkpoints."""

    def __init__(
        self,
        hidden_size: int,
        prior_modeling: str = "mlp_head",
        num_layers: int = 2,
        proj_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden_size = int(hidden_size)
        num_layers = int(num_layers)
        proj_dim = hidden_size if proj_dim is None else int(proj_dim)
        if hidden_size < 1:
            raise ValueError("hidden_size must be greater than 0.")
        if num_layers < 1:
            raise ValueError("num_layers must be greater than 0.")
        if proj_dim < 1:
            raise ValueError("proj_dim must be greater than 0.")

        if prior_modeling == "linear_head":
            self.head = nn.Linear(hidden_size, 1)
        elif prior_modeling == "mlp_head":
            layers: list[nn.Module] = []
            input_dim = hidden_size
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(input_dim, proj_dim))
                layers.append(nn.ReLU())
                input_dim = proj_dim

            layers.append(nn.Linear(input_dim, 1))
            self.head = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown BEFT prior modeling type: {prior_modeling}.")

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except RuntimeError:
            if any(key.startswith("head.") for key in state_dict):
                raise

            prefixed_state = {f"head.{key}": value for key, value in state_dict.items()}
            return super().load_state_dict(prefixed_state, strict=strict, assign=assign)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.head(hidden_states)

