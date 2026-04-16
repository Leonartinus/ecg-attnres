"""Full ECG classifier: tokenizer -> encoder -> pooling -> multi-label head."""
import torch
import torch.nn as nn


class ECGClassifier(nn.Module):
    def __init__(self, tokenizer: nn.Module, encoder: nn.Module,
                 d_model: int, n_classes: int, demographic_dim: int = 0,
                 dropout: float = 0.3):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        # TODO: pooling strategy (CLS token or mean pooling)
        # TODO: 2-layer MLP head: d_model + demographic_dim -> 128 -> n_classes
        raise NotImplementedError

    def forward(self, x: torch.Tensor, demographics: torch.Tensor = None) -> torch.Tensor:
        # x: (batch, n_leads, timesteps)
        # demographics: (batch, demographic_dim) e.g. [age_normalized, sex]
        # returns: (batch, n_classes) raw logits (apply sigmoid in loss)
        raise NotImplementedError
