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
        self.n_leads = n_leads
        self.d_model = d_model

        self.cnn = nn.Sequential(
            nn.Conv1d(1, d_model // 4, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(d_model // 4),
            nn.GELU(),
            nn.Conv1d(d_model // 4, d_model // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # TODO: decide stride pattern that reduces ~1000 timesteps to ~125 per lead
        self.seq_len_per_lead = 1000 // (2 ** 3)
        self.pos_embedding = nn.Embedding(self.seq_len_per_lead, d_model)

        # TODO: add lead embedding if processing leads independently
        self.lead_embedding = nn.Embedding(n_leads, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_leads, timesteps)
        batch_size, n_leads, timesteps = x.shape

        x = x.reshape(batch_size * n_leads, 1, timesteps)  # (batch*n_leads, 1, timesteps)
        x = self.cnn(x)  # (batch*n_leads, d_model, seq_len_per_lead)

        x = x.transpose(1, 2)  # (batch*n_leads, seq_len_per_lead, d_model)
        x = x.reshape(batch_size, n_leads, -1, self.d_model)  # (batch, n_leads, seq_len_per_lead, d_model)

        seq_len = x.shape[2]
        pos_ids = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embedding(pos_ids)  # (seq_len_per_lead, d_model)
        x = x + pos_emb  # (batch, n_leads, seq_len_per_lead, d_model)

        lead_idx = torch.arange(n_leads, device=x.device)
        lead_emb = self.lead_embedding(lead_idx)  # (n_leads, d_model)
        x = x + lead_emb.unsqueeze(1)  # (batch, n_leads, seq_len_per_lead, d_model)

        x = x.reshape(batch_size, n_leads * seq_len, self.d_model) # (batch, seq_len_per_lead, d_model)

        return x
