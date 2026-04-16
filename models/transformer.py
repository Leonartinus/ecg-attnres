"""Standard transformer encoder with configurable pre/post-norm residuals.

Baseline for comparison against AttnRes variants.
"""
import torch
import torch.nn as nn


class StandardTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, prenorm: bool = True):
        super().__init__()
        # TODO: self-attention, feedforward, layer norms
        # TODO: switch between pre-norm and post-norm based on `prenorm` flag
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class StandardTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, prenorm: bool = True):
        super().__init__()
        # TODO: stack n_layers of StandardTransformerLayer
        # TODO: optional final layer norm for prenorm variant
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
