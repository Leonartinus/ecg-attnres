"""1D CNN tokenizer that converts 12-lead ECG signals into token embeddings.

Input:  (batch, 12, 1000)  -- 12 leads, 10 seconds at 100 Hz
Output: (batch, seq_len, d_model)
"""
import torch
import torch.nn as nn


class ECGTokenizer(nn.Module):
    def __init__(self, n_leads: int = 12, d_model: int = 256):
        super().__init__()
        # TODO: 3 Conv1d layers with BatchNorm + GELU, progressively increasing channels
        # TODO: decide stride pattern that reduces ~1000 timesteps to ~125 per lead
        # TODO: add lead embedding if processing leads independently
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_leads, timesteps)
        # returns: (batch, seq_len, d_model)
        raise NotImplementedError
