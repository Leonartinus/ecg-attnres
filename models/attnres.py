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
        # TODO: create one learned query vector per layer: shape (d_model,)
        # TODO: RMSNorm for keys before attention computation
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: maintain hidden_states list; at each layer compute softmax
        #       attention over all previous hidden states using that layer's query
        # TODO: cache alpha weights for analysis (attach to self._last_alphas)
        raise NotImplementedError


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
        self.block_size = n_layers // n_blocks
        # TODO: one learned 1x d_model projection per sublayer (for attn + ffn)
        # TODO: RMSNorm per sublayer
        # TODO: stack transformer sublayers
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: maintain `blocks` list of completed block representations
        # TODO: maintain `partial` running intra-block sum
        # TODO: apply block attention before each sublayer (attn and ffn)
        # TODO: at block boundary, append partial to blocks and reset
        # TODO: cache alpha weights for analysis
        raise NotImplementedError
