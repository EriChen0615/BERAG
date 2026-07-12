# Copyright 2026
"""Compatibility exports for BEFT prior-head modules used by vLLM-BERAG."""

from __future__ import annotations

from beft_prior_head import BeftPriorHead, build_beft_prior_head, build_mlp_prior_head


__all__ = [
    "BeftPriorHead",
    "build_beft_prior_head",
    "build_mlp_prior_head",
]
