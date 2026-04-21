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
        # self.pooling = nn.AdaptiveAvgPool1d(1)

        # TODO: 2-layer MLP head: d_model + demographic_dim -> 128 -> n_classes
        self.head = nn.Sequential(
            nn.Linear(d_model + demographic_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x: torch.Tensor, demographics: torch.Tensor = None) -> torch.Tensor:
        # x: (batch, n_leads, timesteps)
        # demographics: (batch, demographic_dim) e.g. [age_normalized, sex]
        # returns: (batch, n_classes) raw logits (apply sigmoid in loss)
        x = self.tokenizer(x)
        x = self.encoder(x)
        x = x.mean(dim=1) # (batch, d_model) - mean pooling over sequence dimension
        
        if demographics is not None:
            x = torch.cat([x, demographics], dim=-1)
        x = self.head(x)
        return x
