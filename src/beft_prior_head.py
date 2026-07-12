# Copyright 2026
"""Shared BEFT prior-head definitions for training and vLLM inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


def _validate_positive_int(name: str, value: int) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be greater than 0.")
    return value


def build_mlp_prior_head(
    hidden_size: int,
    *,
    num_layers: int = 2,
    proj_dim: int | None = None,
) -> nn.Sequential:
    """Build the BEFT MLP prior head: Linear/ReLU blocks, then a scalar logit."""

    hidden_size = _validate_positive_int("hidden_size", hidden_size)
    num_layers = _validate_positive_int("num_layers", num_layers)
    proj_dim = hidden_size if proj_dim is None else _validate_positive_int("proj_dim", proj_dim)

    layers: list[nn.Module] = []
    input_dim = hidden_size
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(input_dim, proj_dim))
        layers.append(nn.ReLU())
        input_dim = proj_dim

    layers.append(nn.Linear(input_dim, 1))
    return nn.Sequential(*layers)


def build_beft_prior_head(
    hidden_size: int,
    *,
    prior_modeling: str = "mlp_head",
    num_layers: int = 2,
    proj_dim: int | None = None,
) -> nn.Module:
    """Build a BEFT prior head module from the shared architecture definition."""

    hidden_size = _validate_positive_int("hidden_size", hidden_size)
    if prior_modeling == "linear_head":
        return nn.Linear(hidden_size, 1)
    if prior_modeling == "mlp_head":
        return build_mlp_prior_head(
            hidden_size,
            num_layers=num_layers,
            proj_dim=proj_dim,
        )

    raise ValueError(f"Unknown BEFT prior modeling type: {prior_modeling}.")


class BeftPriorHead(nn.Module):
    """vLLM-loadable wrapper around the shared BEFT prior-head architecture."""

    def __init__(
        self,
        hidden_size: int,
        prior_modeling: str = "mlp_head",
        num_layers: int = 2,
        proj_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.head = build_beft_prior_head(
            hidden_size,
            prior_modeling=prior_modeling,
            num_layers=num_layers,
            proj_dim=proj_dim,
        )

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except RuntimeError:
            if any(key.startswith("head.") for key in state_dict):
                raise

            prefixed_state = {f"head.{key}": value for key, value in state_dict.items()}
            return super().load_state_dict(prefixed_state, strict=strict, assign=assign)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.head(hidden_states)


__all__ = [
    "BeftPriorHead",
    "build_beft_prior_head",
    "build_mlp_prior_head",
]
