"""Standard transformer encoder with configurable pre/post-norm residuals.

Baseline for comparison against AttnRes variants.
"""
import torch
import torch.nn as nn


class StandardTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, prenorm: bool = True):
        super().__init__()
        # TODO: self-attention, feedforward, layer norms
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # TODO: switch between pre-norm and post-norm based on `prenorm` flag
        self.prenorm = prenorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prenorm:
            x = x + self._attn_block(self.norm1(x))
            x = x + self.dropout(self.ff(self.norm2(x)))
        else:
            x = self.norm1(x + self._attn_block(x))
            x = self.norm2(x + self.dropout(self.ff(x)))

        return x
    
    def _attn_block(self, x: torch.Tensor) -> torch.Tensor:
        attn, _ = self.attn(x, x, x)
        return self.dropout(attn)


class StandardTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, prenorm: bool = True):
        super().__init__()
        # TODO: stack n_layers of StandardTransformerLayer
        self.layers = nn.ModuleList([
            StandardTransformerLayer(d_model, n_heads, d_ff, dropout, prenorm)
            for _ in range(n_layers)
        ])

        # TODO: optional final layer norm for prenorm variant
        self.final_norm = nn.LayerNorm(d_model) if prenorm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        if self.final_norm is not None:
            x = self.final_norm(x)
            
        return x
