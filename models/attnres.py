"""Attention Residuals (AttnRes) and Block AttnRes encoders.

Reference: Kimi Team (2026), arXiv:2603.15031.
Replaces standard additive residual connections with learned depth-wise attention.
"""
import torch
import torch.nn as nn


class FullAttnResEncoder(nn.Module):
    """Each layer attends over all previous layer outputs via softmax attention.

    h_l = sum_{i=0}^{l-1} alpha_{i->l} * v_i
    where alpha_{i->l} is computed from a learned per-layer query vector w_l.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        # TODO: build n_layers transformer sublayers (attn + ffn)
        self.n_layers = n_layers

        self.attns = nn.ModuleList([nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True) for _ in range(n_layers)])
        self.attn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        self.ffs = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        ) for _ in range(n_layers)])
        self.ff_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # TODO: create one learned query vector per layer: shape (d_model,)
        self.queries = nn.ParameterList([nn.Parameter(torch.zeros(d_model)) for _ in range(2 * n_layers + 1)])

        # TODO: RMSNorm for keys before attention computation
        self.rms_norm = nn.RMSNorm(d_model)
        
        self.final_norm = nn.LayerNorm(d_model)

        self._last_alphas = None  # for caching attention weights

    def _depth_attention(self, query: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute AttnRes: softmax attention over a stack of previous representations.

        Args:
            values: list of tensors, each (batch, seq_len, d_model)
                    — all previous hidden states / layer outputs
            query:  (d_model,) — this layer's learned query vector

        Returns:
            (batch, seq_len, d_model) — weighted combination of values
            (n_values, batch, seq_len) — the alpha weights (for analysis)
        """
        V = torch.stack(values, dim=0)  # (n_values, batch, seq_len, d_model)
        K = self.rms_norm(V)  # (n_values, batch, seq_len, d_model)

        logits = torch.einsum('d, n b t d -> n b t', query, K)  # (n_values, batch, seq_len)

        alphas = torch.softmax(logits, dim=0)  # (n_values, batch, seq_len)

        h = torch.einsum('n b t, n b t d -> b t d', alphas, V)  # (batch, seq_len, d_model)

        return h, alphas

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: maintain hidden_states list; at each layer compute softmax
        #       attention over all previous hidden states using that layer's query
        all_alphas = []
        hidden_states = [x]

        for l in range(self.n_layers):
            # Compute AttnRes for this layer
            h, alphas = self._depth_attention(self.queries[2*l], hidden_states)
            all_alphas.append(alphas)

            # Apply self-attention sublayer with h as input
            normed = self.attn_norms[l](h)
            attn_out, _ = self.attns[l](normed, normed, normed)

            hidden_states.append(attn_out)

            # Apply feedforward sublayer
            h2, alphas_ff = self._depth_attention(self.queries[2*l + 1], hidden_states)
            all_alphas.append(alphas_ff)

            ff_out = self.ffs[l](self.ff_norms[l](h2))

            hidden_states.append(ff_out)

        output, alpha_final = self._depth_attention(self.queries[-1], hidden_states)
        all_alphas.append(alpha_final)

        # TODO: cache alpha weights for analysis (attach to self._last_alphas)
        self._last_alphas = all_alphas

        return self.final_norm(output)


class BlockAttnResEncoder(nn.Module):
    """Partitions layers into N blocks; standard residuals within a block,
    softmax attention over block-level representations between blocks.

    More memory-efficient than FullAttnResEncoder while preserving most gains.
    Recommended default: n_blocks = 3 for a 6-layer transformer.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int,
                 n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        assert n_layers % n_blocks == 0, "n_layers must be divisible by n_blocks"
        self.n_layers = n_layers
        self.n_blocks = n_blocks
        self.block_size = n_layers // n_blocks

        # TODO: one learned 1x d_model projection per sublayer (for attn + ffn)
        self.queries = nn.ParameterList([
            nn.Parameter(torch.zeros(d_model)) for _ in range(2 * n_layers + 1)
        ])

        # TODO: RMSNorm per sublayer
        self.depth_norm = nn.RMSNorm(d_model)

        # TODO: stack transformer sublayers
        self.attns = self.attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.attn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.attn_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        self.ff_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        self.final_norm = nn.LayerNorm(d_model)

        self._last_alphas = None  # for caching attention weights

    def _block_attention(self, query: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute block-level AttnRes: softmax attention over block representations.

        Args:
            values: list of tensors, each (batch, seq_len, d_model)
                    — all previous block outputs
            query:  (d_model,) — this block's learned query vector

        Returns:
            (batch, seq_len, d_model) — weighted combination of block outputs
            (n_values, batch, seq_len) — the alpha weights (for analysis)
        """
        V = torch.stack(values, dim=0)  # (n_values, batch, seq_len, d_model)
        K = self.depth_norm(V)  # (n_values, batch, seq_len, d_model)

        logits = torch.einsum('d, n b t d -> n b t', query, K)  # (n_values, batch, seq_len)

        alphas = torch.softmax(logits, dim=0)  # (n_values, batch, seq_len)

        h = torch.einsum('n b t, n b t d -> b t d', alphas, V)  # (batch, seq_len, d_model)

        return h, alphas

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: maintain `blocks` list of completed block representations
        blocks = [x]

        # TODO: maintain `partial` running intra-block sum
        partial = x

        all_alphas = []

        # TODO: apply block attention before each sublayer (attn and ffn)
        for i in range(self.n_layers):
            values = blocks + [partial]  # Attend over completed blocks + current partial
            # Apply block attention to get block-level residual
            h, attn_alphas = self._block_attention(self.queries[2 * i], values)
            all_alphas.append(attn_alphas)  # Cache attention weights

            normed = self.attn_norms[i](h)
            attn_out, _ = self.attns[i](normed, normed, normed)
            attn_out = self.attn_dropouts[i](attn_out)

            # Apply the i-th sublayer with block_res as input
            partial = partial + attn_out  # Add block-level residual

            values = blocks + [partial]  # Attend over completed blocks + current partial
            h2, ff_alphas = self._block_attention(self.queries[2 * i + 1], values)
            all_alphas.append(ff_alphas)  # Cache attention weights

            ff_out = self.ffns[i](self.ff_norms[i](h2))
            # Apply the i-th FFN with block-level residual
            partial = partial + ff_out  # Add block-level residual

            # TODO: at block boundary, append partial to blocks and reset partial
            if (i + 1) % self.block_size == 0:
                blocks.append(partial)
                partial = torch.zeros_like(x)
        
        values = blocks + [partial]
        output, alpha_final = self._block_attention(self.queries[-1], values)
        all_alphas.append(alpha_final)

        # TODO: cache alpha weights for analysis
        self._last_alphas = all_alphas
        return self.final_norm(output)
