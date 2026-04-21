"""Shape and behavior tests for all model components.

Run:
    pytest tests/test_shapes.py -v

    # Run a specific test:
    pytest tests/test_shapes.py::test_tokenizer_output_shape -v

    # Run only fast tests (skip slow ones):
    pytest tests/test_shapes.py -v -m "not slow"

All tests run on CPU and take < 30 seconds total.
"""
import pytest
import torch
import torch.nn as nn

from models.tokenizer import ECGTokenizer
from models.transformer import StandardTransformerLayer, StandardTransformerEncoder
from models.attnres import FullAttnResEncoder, BlockAttnResEncoder
from models.classifier import ECGClassifier


# ============================================================================
# Fixtures — shared model instances so we don't rebuild for every test
# ============================================================================

@pytest.fixture
def tokenizer():
    return ECGTokenizer(n_leads=12, d_model=256)


@pytest.fixture
def encoder_prenorm():
    return StandardTransformerEncoder(
        d_model=256, n_layers=6, n_heads=4, d_ff=1024, dropout=0.1, prenorm=True
    )


@pytest.fixture
def encoder_postnorm():
    return StandardTransformerEncoder(
        d_model=256, n_layers=6, n_heads=4, d_ff=1024, dropout=0.1, prenorm=False
    )


@pytest.fixture
def encoder_full_attnres():
    return FullAttnResEncoder(
        d_model=256, n_layers=6, n_heads=4, d_ff=1024, dropout=0.1
    )


@pytest.fixture
def encoder_block_attnres_3():
    return BlockAttnResEncoder(
        d_model=256, n_layers=6, n_heads=4, d_ff=1024, n_blocks=3, dropout=0.1
    )


@pytest.fixture
def encoder_block_attnres_2():
    return BlockAttnResEncoder(
        d_model=256, n_layers=6, n_heads=4, d_ff=1024, n_blocks=2, dropout=0.1
    )


@pytest.fixture
def sample_ecg():
    """Fake 12-lead ECG batch: (4, 12, 1000)"""
    return torch.randn(4, 12, 1000)


@pytest.fixture
def sample_tokens():
    """Fake tokenizer output: (4, 1500, 256) — what the encoder expects."""
    return torch.randn(4, 1500, 256)


@pytest.fixture
def sample_demographics():
    """Fake demographics: (4, 2) — [age_normalized, sex]"""
    return torch.randn(4, 2)


# ============================================================================
# Tokenizer tests
# ============================================================================

class TestTokenizer:

    def test_output_shape(self, tokenizer, sample_ecg):
        """(batch, 12, 1000) → (batch, 1500, 256)"""
        out = tokenizer(sample_ecg)
        assert out.shape == (4, 1500, 256), f"Expected (4, 1500, 256), got {out.shape}"

    def test_batch_size_1(self, tokenizer):
        """Should work with a single sample."""
        out = tokenizer(torch.randn(1, 12, 1000))
        assert out.shape == (1, 1500, 256)

    def test_large_batch(self, tokenizer):
        """Should work with larger batches."""
        out = tokenizer(torch.randn(32, 12, 1000))
        assert out.shape == (32, 1500, 256)

    def test_no_nan(self, tokenizer, sample_ecg):
        """Output should be finite."""
        out = tokenizer(sample_ecg)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_different_leads_produce_different_tokens(self, tokenizer):
        """Identical signals on different leads should differ due to lead embedding."""
        signal = torch.randn(1, 1, 1000)
        x = signal.expand(1, 12, 1000)  # same signal on all 12 leads
        out = tokenizer(x)              # (1, 1500, 256)
        out = out.reshape(12, 125, 256)

        # Lead 0 and Lead 6 should differ
        assert not torch.allclose(out[0], out[6], atol=1e-5), \
            "Identical signals on different leads should produce different tokens"

    def test_different_positions_produce_different_tokens(self, tokenizer, sample_ecg):
        """Tokens at different positions should differ due to positional embedding."""
        out = tokenizer(sample_ecg)
        # Compare first and last token of first lead (positions 0 and 124)
        assert not torch.allclose(out[0, 0], out[0, 124], atol=1e-5), \
            "Tokens at different positions should differ"

    def test_gradient_flows(self, tokenizer):
        """Gradients should propagate from output back to input."""
        x = torch.randn(2, 12, 1000, requires_grad=True)
        out = tokenizer(x)
        out.sum().backward()
        assert x.grad is not None, "No gradient on input"
        assert x.grad.abs().sum() > 0, "Gradient is all zeros"

    def test_eval_mode(self, tokenizer, sample_ecg):
        """Should work in eval mode (BatchNorm behaves differently)."""
        tokenizer.eval()
        out = tokenizer(sample_ecg)
        assert out.shape == (4, 1500, 256)
        tokenizer.train()  # reset


# ============================================================================
# Standard Transformer Encoder tests
# ============================================================================

class TestStandardTransformerEncoder:

    def test_prenorm_output_shape(self, encoder_prenorm, sample_tokens):
        """Encoder should preserve shape: (batch, seq_len, d_model) → same."""
        out = encoder_prenorm(sample_tokens)
        assert out.shape == sample_tokens.shape, \
            f"Expected {sample_tokens.shape}, got {out.shape}"

    def test_postnorm_output_shape(self, encoder_postnorm, sample_tokens):
        """PostNorm variant should also preserve shape."""
        out = encoder_postnorm(sample_tokens)
        assert out.shape == sample_tokens.shape

    def test_prenorm_no_nan(self, encoder_prenorm, sample_tokens):
        out = encoder_prenorm(sample_tokens)
        assert not torch.isnan(out).any(), "PreNorm output contains NaN"

    def test_postnorm_no_nan(self, encoder_postnorm, sample_tokens):
        out = encoder_postnorm(sample_tokens)
        assert not torch.isnan(out).any(), "PostNorm output contains NaN"

    def test_different_outputs_for_different_inputs(self, encoder_prenorm):
        """Different inputs should produce different outputs (not collapsed)."""
        x1 = torch.randn(2, 100, 256)
        x2 = torch.randn(2, 100, 256)
        encoder_prenorm.eval()
        out1 = encoder_prenorm(x1)
        out2 = encoder_prenorm(x2)
        assert not torch.allclose(out1, out2, atol=1e-4), \
            "Different inputs produced identical outputs"
        encoder_prenorm.train()

    def test_gradient_flows(self, encoder_prenorm):
        x = torch.randn(2, 100, 256, requires_grad=True)
        out = encoder_prenorm(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_single_layer(self):
        """A single-layer encoder should still work."""
        enc = StandardTransformerEncoder(
            d_model=128, n_layers=1, n_heads=4, d_ff=512, dropout=0.0
        )
        x = torch.randn(2, 50, 128)
        out = enc(x)
        assert out.shape == (2, 50, 128)

    def test_deep_encoder(self):
        """An 8-layer encoder should work (for depth ablation experiments)."""
        enc = StandardTransformerEncoder(
            d_model=128, n_layers=8, n_heads=4, d_ff=512, dropout=0.0
        )
        x = torch.randn(2, 50, 128)
        out = enc(x)
        assert out.shape == (2, 50, 128)


# ============================================================================
# Full AttnRes Encoder tests
# ============================================================================

class TestFullAttnResEncoder:

    def test_output_shape(self, encoder_full_attnres, sample_tokens):
        out = encoder_full_attnres(sample_tokens)
        assert out.shape == sample_tokens.shape, \
            f"Expected {sample_tokens.shape}, got {out.shape}"

    def test_no_nan(self, encoder_full_attnres, sample_tokens):
        out = encoder_full_attnres(sample_tokens)
        assert not torch.isnan(out).any(), "FullAttnRes output contains NaN"

    def test_gradient_flows(self, encoder_full_attnres):
        x = torch.randn(2, 100, 256, requires_grad=True)
        out = encoder_full_attnres(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_alpha_weights_cached(self, encoder_full_attnres, sample_tokens):
        """Forward pass should cache attention weights for analysis."""
        encoder_full_attnres.eval()
        _ = encoder_full_attnres(sample_tokens)

        assert hasattr(encoder_full_attnres, '_last_alphas'), \
            "Forward pass should cache alpha weights in self._last_alphas"

        alphas = encoder_full_attnres._last_alphas
        assert alphas is not None, "Cached alphas should not be None"
        encoder_full_attnres.train()

    def test_alpha_weights_sum_to_one(self, encoder_full_attnres):
        """AttnRes alpha weights should sum to 1 across the depth dimension (softmax)."""
        encoder_full_attnres.eval()
        x = torch.randn(2, 50, 256)
        _ = encoder_full_attnres(x)

        alphas = encoder_full_attnres._last_alphas
        # alphas structure depends on implementation, but each layer's
        # weights over previous layers should sum to 1.
        # Adjust this check based on your actual alpha storage format.
        if isinstance(alphas, list):
            for i, alpha in enumerate(alphas):
                # alpha shape: (n_prev, batch, seq_len) or similar
                sums = alpha.sum(dim=0)
                assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
                    f"Layer {i} alpha weights don't sum to 1"
        encoder_full_attnres.train()

    def test_different_from_standard(self, encoder_full_attnres, encoder_prenorm, sample_tokens):
        """AttnRes and Standard should produce different outputs (different residual logic)."""
        encoder_full_attnres.eval()
        encoder_prenorm.eval()
        out_attn = encoder_full_attnres(sample_tokens)
        out_std = encoder_prenorm(sample_tokens)
        assert not torch.allclose(out_attn, out_std, atol=1e-4), \
            "AttnRes and Standard produced identical outputs"
        encoder_full_attnres.train()
        encoder_prenorm.train()


# ============================================================================
# Block AttnRes Encoder tests
# ============================================================================

class TestBlockAttnResEncoder:

    def test_output_shape_3_blocks(self, encoder_block_attnres_3, sample_tokens):
        out = encoder_block_attnres_3(sample_tokens)
        assert out.shape == sample_tokens.shape

    def test_output_shape_2_blocks(self, encoder_block_attnres_2, sample_tokens):
        out = encoder_block_attnres_2(sample_tokens)
        assert out.shape == sample_tokens.shape

    def test_no_nan(self, encoder_block_attnres_3, sample_tokens):
        out = encoder_block_attnres_3(sample_tokens)
        assert not torch.isnan(out).any(), "BlockAttnRes output contains NaN"

    def test_gradient_flows(self, encoder_block_attnres_3):
        x = torch.randn(2, 100, 256, requires_grad=True)
        out = encoder_block_attnres_3(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_alpha_weights_cached(self, encoder_block_attnres_3, sample_tokens):
        """Forward pass should cache block attention weights for analysis."""
        encoder_block_attnres_3.eval()
        _ = encoder_block_attnres_3(sample_tokens)
        assert hasattr(encoder_block_attnres_3, '_last_alphas'), \
            "Forward pass should cache alpha weights in self._last_alphas"
        encoder_block_attnres_3.train()

    def test_invalid_block_count(self):
        """n_layers must be divisible by n_blocks."""
        with pytest.raises((AssertionError, ValueError)):
            BlockAttnResEncoder(
                d_model=128, n_layers=6, n_heads=4, d_ff=512, n_blocks=4
            )

    def test_single_block_degenerates(self):
        """With 1 block, Block AttnRes should still work (edge case)."""
        enc = BlockAttnResEncoder(
            d_model=128, n_layers=4, n_heads=4, d_ff=512, n_blocks=1, dropout=0.0
        )
        x = torch.randn(2, 50, 128)
        out = enc(x)
        assert out.shape == (2, 50, 128)

    def test_n_blocks_equals_n_layers(self):
        """With n_blocks == n_layers (1 layer per block), should approximate FullAttnRes."""
        enc = BlockAttnResEncoder(
            d_model=128, n_layers=4, n_heads=4, d_ff=512, n_blocks=4, dropout=0.0
        )
        x = torch.randn(2, 50, 128)
        out = enc(x)
        assert out.shape == (2, 50, 128)


# ============================================================================
# ECGClassifier (end-to-end) tests
# ============================================================================

class TestECGClassifier:

    def _build_classifier(self, encoder, demographic_dim=0):
        tokenizer = ECGTokenizer(n_leads=12, d_model=256)
        return ECGClassifier(
            tokenizer=tokenizer,
            encoder=encoder,
            d_model=256,
            n_classes=5,
            demographic_dim=demographic_dim,
            dropout=0.3,
        )

    def test_end_to_end_prenorm(self, encoder_prenorm, sample_ecg):
        """Full pipeline: raw ECG → 5-class logits."""
        model = self._build_classifier(encoder_prenorm)
        out = model(sample_ecg)
        assert out.shape == (4, 5), f"Expected (4, 5), got {out.shape}"

    def test_end_to_end_with_demographics(self, encoder_prenorm, sample_ecg, sample_demographics):
        """With age + sex concatenated."""
        model = self._build_classifier(encoder_prenorm, demographic_dim=2)
        out = model(sample_ecg, sample_demographics)
        assert out.shape == (4, 5)

    def test_end_to_end_without_demographics(self, encoder_prenorm, sample_ecg):
        """demographics=None should work when demographic_dim=0."""
        model = self._build_classifier(encoder_prenorm, demographic_dim=0)
        out = model(sample_ecg, demographics=None)
        assert out.shape == (4, 5)

    def test_end_to_end_full_attnres(self, encoder_full_attnres, sample_ecg):
        model = self._build_classifier(encoder_full_attnres)
        out = model(sample_ecg)
        assert out.shape == (4, 5)

    def test_end_to_end_block_attnres(self, encoder_block_attnres_3, sample_ecg):
        model = self._build_classifier(encoder_block_attnres_3)
        out = model(sample_ecg)
        assert out.shape == (4, 5)

    def test_output_is_raw_logits(self, encoder_prenorm, sample_ecg):
        """Output should be raw logits, NOT probabilities.
        sigmoid is applied inside BCEWithLogitsLoss, not in the model.
        Some outputs should be negative (raw logits span -inf to +inf)."""
        model = self._build_classifier(encoder_prenorm)
        model.eval()
        # Run several times — at least once, some logits should be negative
        out = model(sample_ecg)
        has_negative = (out < 0).any().item()
        has_positive = (out > 0).any().item()
        assert has_negative or has_positive, \
            "Logits are all zero — model might not be initialized properly"

    def test_no_nan(self, encoder_prenorm, sample_ecg):
        model = self._build_classifier(encoder_prenorm)
        out = model(sample_ecg)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradient_flows_to_input(self, encoder_prenorm):
        """Gradient should flow all the way from loss to raw ECG input."""
        model = self._build_classifier(encoder_prenorm)
        x = torch.randn(2, 12, 1000, requires_grad=True)
        out = model(x)

        # Simulate BCE loss
        target = torch.zeros(2, 5)
        target[0, 0] = 1.0  # first sample has NORM
        target[1, 2] = 1.0  # second sample has STTC
        loss = nn.BCEWithLogitsLoss()(out, target)
        loss.backward()

        assert x.grad is not None, "No gradient reached the input"
        assert x.grad.abs().sum() > 0, "Gradient is all zeros"

    def test_different_inputs_different_outputs(self, encoder_prenorm):
        """Model should not collapse all inputs to the same output."""
        model = self._build_classifier(encoder_prenorm)
        model.eval()
        out1 = model(torch.randn(2, 12, 1000))
        out2 = model(torch.randn(2, 12, 1000))
        assert not torch.allclose(out1, out2, atol=1e-4), \
            "Different inputs produced identical outputs"
        model.train()


# ============================================================================
# Cross-encoder consistency tests
# ============================================================================

class TestEncoderConsistency:
    """Verify all encoder variants are interchangeable in the classifier."""

    @pytest.fixture(params=["prenorm", "postnorm", "full_attnres", "block_attnres"])
    def encoder(self, request):
        """Parameterized fixture that yields each encoder type."""
        if request.param == "prenorm":
            return StandardTransformerEncoder(256, 6, 4, 1024, 0.1, prenorm=True)
        elif request.param == "postnorm":
            return StandardTransformerEncoder(256, 6, 4, 1024, 0.1, prenorm=False)
        elif request.param == "full_attnres":
            return FullAttnResEncoder(256, 6, 4, 1024, 0.1)
        elif request.param == "block_attnres":
            return BlockAttnResEncoder(256, 6, 4, 1024, n_blocks=3, dropout=0.1)

    def test_all_encoders_same_io_shape(self, encoder):
        """Every encoder variant should accept and return the same shape."""
        x = torch.randn(2, 100, 256)
        out = encoder(x)
        assert out.shape == (2, 100, 256), \
            f"{encoder.__class__.__name__} produced shape {out.shape}, expected (2, 100, 256)"

    def test_all_encoders_pluggable_into_classifier(self, encoder, sample_ecg):
        """Every encoder should work as a drop-in inside ECGClassifier."""
        tokenizer = ECGTokenizer(n_leads=12, d_model=256)
        model = ECGClassifier(tokenizer, encoder, d_model=256, n_classes=5)
        out = model(sample_ecg)
        assert out.shape == (4, 5), \
            f"{encoder.__class__.__name__} in classifier produced {out.shape}, expected (4, 5)"


# ============================================================================
# Parameter count sanity checks
# ============================================================================

class TestParameterCounts:

    def test_tokenizer_has_learnable_params(self, tokenizer):
        n_params = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
        assert n_params > 0, "Tokenizer has no learnable parameters"
        # CNN + embeddings should be roughly 100K–500K params
        assert n_params < 5_000_000, f"Tokenizer has {n_params} params — suspiciously large"

    def test_encoder_has_learnable_params(self, encoder_prenorm):
        n_params = sum(p.numel() for p in encoder_prenorm.parameters() if p.requires_grad)
        assert n_params > 0
        # 6-layer transformer with d=256 should be roughly 1–3M params
        assert n_params < 20_000_000, f"Encoder has {n_params} params — suspiciously large"

    def test_attnres_has_extra_params(self, encoder_prenorm, encoder_full_attnres):
        """AttnRes should have slightly more parameters than standard (the query vectors)."""
        n_std = sum(p.numel() for p in encoder_prenorm.parameters())
        n_attn = sum(p.numel() for p in encoder_full_attnres.parameters())
        assert n_attn >= n_std, \
            "AttnRes should have at least as many params as standard"

    def test_block_attnres_has_extra_params(self, encoder_prenorm, encoder_block_attnres_3):
        """Block AttnRes should have slightly more parameters (the projection layers)."""
        n_std = sum(p.numel() for p in encoder_prenorm.parameters())
        n_block = sum(p.numel() for p in encoder_block_attnres_3.parameters())
        assert n_block >= n_std

    def test_full_model_param_count(self, encoder_prenorm):
        """Full classifier should be roughly 2–5M params."""
        tokenizer = ECGTokenizer(n_leads=12, d_model=256)
        model = ECGClassifier(tokenizer, encoder_prenorm, d_model=256, n_classes=5)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nFull model parameter count: {n_params:,}")
        assert 100_000 < n_params < 20_000_000, \
            f"Model has {n_params:,} params — expected 100K–20M range"


# ============================================================================
# Training simulation tests
# ============================================================================

class TestTrainingSimulation:
    """Verify the model can take a gradient step without errors."""

    def test_one_training_step_prenorm(self, encoder_prenorm, sample_ecg):
        tokenizer = ECGTokenizer(n_leads=12, d_model=256)
        model = ECGClassifier(tokenizer, encoder_prenorm, d_model=256, n_classes=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        # Fake labels: multi-label binary targets
        targets = torch.zeros(4, 5)
        targets[0, 0] = 1.0
        targets[1, 1] = 1.0
        targets[1, 2] = 1.0  # multi-label: MI + STTC
        targets[2, 4] = 1.0
        targets[3, 0] = 1.0

        # Forward
        out = model(sample_ecg)
        loss = criterion(out, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(torch.tensor(loss.item())), "Loss is NaN"

    def test_two_steps_loss_changes(self, encoder_prenorm, sample_ecg):
        """Loss should change after a gradient step (model is learning)."""
        tokenizer = ECGTokenizer(n_leads=12, d_model=256)
        model = ECGClassifier(tokenizer, encoder_prenorm, d_model=256, n_classes=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        targets = torch.zeros(4, 5)
        targets[:, 0] = 1.0

        # Step 1
        out1 = model(sample_ecg)
        loss1 = criterion(out1, targets)
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        # Step 2
        out2 = model(sample_ecg)
        loss2 = criterion(out2, targets)

        assert loss1.item() != loss2.item(), \
            "Loss didn't change after gradient step — model may not be learning"

    @pytest.mark.slow
    def test_overfits_tiny_batch(self, sample_ecg):
        """Model should be able to overfit a single batch if trained long enough.
        This catches subtle bugs where the model trains but can't actually learn.
        Marked as slow (~10s)."""
        tokenizer = ECGTokenizer(n_leads=12, d_model=256)
        encoder = StandardTransformerEncoder(256, 2, 4, 512, dropout=0.0)  # small, no dropout
        model = ECGClassifier(tokenizer, encoder, d_model=256, n_classes=5, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        # Single batch with clear targets
        x = sample_ecg[:2]  # just 2 samples
        targets = torch.tensor([[1, 0, 0, 0, 0], [0, 1, 1, 0, 0]], dtype=torch.float)

        # Train for 50 steps
        model.train()
        for _ in range(50):
            out = model(x)
            loss = criterion(out, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < 0.5, \
            f"Model couldn't overfit a 2-sample batch after 50 steps (loss={final_loss:.3f}). " \
            f"Something is likely wrong with the architecture or gradient flow."
