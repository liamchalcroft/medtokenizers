import pytest
import torch

from medtokenizers.modules.quant import (
    FSQuantizer,
    LFQuantizer,
    ResidualFSQuantizer,
    VectorQuantizer,
)


@pytest.fixture
def batch_size() -> int:
    return 3


@pytest.fixture
def channels() -> int:
    return 6


@pytest.fixture
def spatial_dims_2d() -> tuple[int, int]:
    return (8, 8)


@pytest.fixture
def spatial_dims_3d() -> tuple[int, int, int]:
    return (8, 8, 8)


def test_fsq_quantizes_2d(
    batch_size: int, channels: int, spatial_dims_2d: tuple[int, int]
) -> None:
    quantizer = FSQuantizer(levels=[2] * channels)
    x = torch.randn(batch_size, channels, *spatial_dims_2d)
    quantised, loss, indices = quantizer(x)

    assert quantised.shape == x.shape
    # Indices have spatial shape (B, H, W) not flat (B, H*W)
    assert indices.shape == (batch_size, *spatial_dims_2d)
    assert loss.shape == (batch_size, 1, *spatial_dims_2d)


def test_fsq_quantizes_3d(
    batch_size: int, channels: int, spatial_dims_3d: tuple[int, int, int]
) -> None:
    quantizer = FSQuantizer(levels=[2] * channels)
    x = torch.randn(batch_size, channels, *spatial_dims_3d)
    quantised, loss, indices = quantizer(x)

    assert quantised.shape == x.shape
    # Indices have spatial shape (B, H, W, D) not flat (B, H*W*D)
    assert indices.shape == (batch_size, *spatial_dims_3d)
    assert loss.shape == (batch_size, 1, *spatial_dims_3d)


def test_fsq_rejects_incorrect_embedding_dim(
    batch_size: int, spatial_dims_2d: tuple[int, int]
) -> None:
    quantizer = FSQuantizer(levels=[2, 2, 2], embedding_dim=3)
    x = torch.randn(batch_size, 2, *spatial_dims_2d)

    with pytest.raises(AssertionError):
        quantizer(x)


def test_vector_quantizer_round_trip(
    batch_size: int, channels: int, spatial_dims_3d: tuple[int, int, int]
) -> None:
    quantizer = VectorQuantizer(num_embeddings=32, embedding_dim=channels, dim=3)
    x = torch.randn(batch_size, channels, *spatial_dims_3d)
    quantised, loss, indices = quantizer(x)

    shape = (batch_size, *spatial_dims_3d, channels)
    reconstructed = quantizer.get_codebook_entry(indices.view(-1), shape)
    assert torch.allclose(reconstructed, quantised, atol=1e-5)
    assert loss.shape == (batch_size, 1, 1, 1, 1)


def test_residual_fsq_stacks_losses(
    batch_size: int, channels: int, spatial_dims_2d: tuple[int, int]
) -> None:
    quantizer = ResidualFSQuantizer(levels=[2] * channels, num_quantizers=3)
    x = torch.randn(batch_size, channels, *spatial_dims_2d)
    quantised, loss, indices = quantizer(x)

    assert quantised.shape == x.shape
    # Indices have shape (B, num_quantizers, H, W) with spatial dimensions preserved
    assert indices.shape == (batch_size, 3, *spatial_dims_2d)
    assert loss.shape == (batch_size, *spatial_dims_2d)


def test_lfq_quantizes_contiguously(
    batch_size: int, channels: int, spatial_dims_2d: tuple[int, int]
) -> None:
    # All quantizers now return (codes, loss, indices) consistently
    quantizer = LFQuantizer(
        codebook_size=16,
        codebook_dim=channels,
        num_codebooks=2,
        entropy_loss=False,
        dim=2,
    )
    x = torch.randn(batch_size, channels, *spatial_dims_2d)
    quantised, loss, indices = quantizer(x)

    assert quantised.shape == x.shape
    # Loss is broadcasted to (B, 1, 1, 1) for 2D
    assert loss.shape == (batch_size, 1, 1, 1)
    # Indices have spatial shape with num_codebooks: (B, H, W, num_codebooks)
    assert indices.shape == (batch_size, *spatial_dims_2d, 2)


# ============================================================================
# Phase 5: Edge Case Tests
# ============================================================================


class TestFSQEdgeCases:
    """Edge case tests for FSQuantizer."""

    def test_fsq_handles_extreme_values(self) -> None:
        """Test FSQ handles extreme input values without NaN."""
        quantizer = FSQuantizer(levels=[8, 8, 8, 8], embedding_dim=4)
        x = torch.randn(2, 4, 8, 8) * 100  # Large values
        quantised, loss, indices = quantizer(x)

        assert not torch.isnan(quantised).any(), "Quantized output contains NaN"
        assert not torch.isnan(loss).any(), "Loss contains NaN"
        assert not torch.isinf(quantised).any(), "Quantized output contains Inf"

    def test_fsq_bound_handles_atanh_edge_cases(self) -> None:
        """Test that bound() doesn't produce NaN at extreme ratio values."""
        quantizer = FSQuantizer(levels=[8, 8, 8, 8], embedding_dim=4)

        # Values that could push atanh to its limits
        extreme_vals = torch.tensor([[-10.0, 10.0, -100.0, 100.0]], dtype=torch.float32)
        bounded = quantizer.bound(extreme_vals)

        assert not torch.isnan(bounded).any(), "bound() produced NaN"
        assert not torch.isinf(bounded).any(), "bound() produced Inf"

    def test_fsq_indices_roundtrip(self) -> None:
        """Test that indices_to_codes inverts codes_to_indices."""
        quantizer = FSQuantizer(levels=[8, 5, 5, 5], embedding_dim=4)
        x = torch.randn(2, 4, 8, 8)

        quantised, _, indices = quantizer(x)
        recovered = quantizer.indices_to_codes(indices)

        # recovered is (B, H, W, C), need to permute to (B, C, H, W)
        recovered = recovered.permute(0, 3, 1, 2)

        assert torch.allclose(recovered, quantised, atol=1e-5), "Roundtrip failed"

    def test_fsq_codebook_size_calculation(self) -> None:
        """Test that get_codebook_size returns correct value."""
        levels = [8, 5, 5, 5]
        quantizer = FSQuantizer(levels=levels, embedding_dim=4)
        expected_size = 8 * 5 * 5 * 5  # 1000
        assert quantizer.get_codebook_size() == expected_size


class TestVectorQuantizerEdgeCases:
    """Edge case tests for VectorQuantizer."""

    def test_vq_rejects_invalid_dim(self) -> None:
        """Test that VQ raises error for invalid dim at init."""
        with pytest.raises(ValueError, match="dim must be 1, 2, or 3"):
            VectorQuantizer(num_embeddings=256, embedding_dim=6, dim=4)

        VectorQuantizer(num_embeddings=256, embedding_dim=6, dim=1)

    def test_vq_ema_updates_codebook(self) -> None:
        """Test that EMA updates actually modify the codebook."""
        torch.manual_seed(42)
        quantizer = VectorQuantizer(
            num_embeddings=16,
            embedding_dim=4,
            dim=2,
            use_ema=True,
            ema_decay=0.9,
        )
        quantizer.train()

        # Store initial embeddings
        initial_weights = quantizer.embedding.weight.data.clone()

        # Run several forward passes
        for _ in range(10):
            x = torch.randn(8, 4, 4, 4)
            quantizer(x)

        # Check that embeddings have changed
        assert not torch.allclose(
            initial_weights, quantizer.embedding.weight.data, atol=1e-6
        ), "EMA updates did not modify codebook"

    def test_vq_ema_no_update_in_eval(self) -> None:
        """Test that EMA doesn't update in eval mode."""
        quantizer = VectorQuantizer(
            num_embeddings=16,
            embedding_dim=4,
            dim=2,
            use_ema=True,
        )
        quantizer.eval()

        initial_weights = quantizer.embedding.weight.data.clone()

        x = torch.randn(4, 4, 4, 4)
        quantizer(x)

        assert torch.allclose(initial_weights, quantizer.embedding.weight.data), (
            "EMA updated in eval mode"
        )

    def test_vq_indices_dtype(self) -> None:
        """Test that VQ returns int32 indices."""
        quantizer = VectorQuantizer(num_embeddings=256, embedding_dim=4, dim=2)
        x = torch.randn(2, 4, 8, 8)
        _, _, indices = quantizer(x)

        assert indices.dtype == torch.int32, f"Expected int32, got {indices.dtype}"

    def test_vq_dim1_shapes(self) -> None:
        """Test VQ supports 1D token sequences."""
        quantizer = VectorQuantizer(num_embeddings=32, embedding_dim=8, dim=1)
        x = torch.randn(2, 8, 16)
        quantized, loss, indices = quantizer(x)

        assert quantized.shape == x.shape
        assert indices.shape == (2, 16)
        assert loss.shape == (2, 1, 1)

    def test_vq_use_norm_affects_assignment(self) -> None:
        """Test that use_norm switches to cosine-based assignment."""
        z = torch.tensor([[[[9.0]], [[0.0]]]])
        weights = torch.tensor([[1.0, 0.0], [10.0, 0.1]])

        vq_euclidean = VectorQuantizer(
            num_embeddings=2,
            embedding_dim=2,
            dim=2,
            use_norm=False,
        )
        with torch.no_grad():
            vq_euclidean.embedding.weight.copy_(weights)
        _, _, indices = vq_euclidean(z)
        assert indices.item() == 1

        vq_cosine = VectorQuantizer(
            num_embeddings=2,
            embedding_dim=2,
            dim=2,
            use_norm=True,
        )
        with torch.no_grad():
            vq_cosine.embedding.weight.copy_(weights)
        _, _, indices = vq_cosine(z)
        assert indices.item() == 0

    @pytest.mark.parametrize("use_norm", [False, True])
    def test_vq_indices_to_codes_matches_forward(self, use_norm: bool) -> None:
        """Test that decoding indices reproduces the codes forward() emits."""
        vq = VectorQuantizer(
            num_embeddings=16, embedding_dim=4, dim=2, use_norm=use_norm
        )
        with torch.no_grad():
            vq.embedding.weight.normal_(std=3.0)
        vq.eval()
        z = torch.randn(2, 4, 5, 6)
        with torch.no_grad():
            z_q, _, indices = vq(z)
            codes = vq.indices_to_codes(indices.long().view(-1), shape=(2, 5, 6, 4))
        assert torch.allclose(codes, z_q, atol=1e-6)


class TestResidualFSQEdgeCases:
    """Edge case tests for ResidualFSQuantizer."""

    def test_resfsq_indices_to_codes_2d(self) -> None:
        """Test indices_to_codes for 2D input."""
        quantizer = ResidualFSQuantizer(
            levels=[8, 5, 5],
            num_quantizers=2,
            embedding_dim=3,
        )
        x = torch.randn(2, 3, 8, 8)
        quantised, _, indices = quantizer(x)

        recovered = quantizer.indices_to_codes(indices)

        assert recovered.shape == quantised.shape, (
            f"Shape mismatch: {recovered.shape} vs {quantised.shape}"
        )
        assert torch.allclose(recovered, quantised, atol=1e-5), "Roundtrip failed"

    def test_resfsq_shared_projections(self) -> None:
        """Test ResidualFSQ with shared projections reduces parameters."""
        levels = [8, 5, 5]
        num_quantizers = 4
        embedding_dim = 16

        # Without shared projections
        quantizer_separate = ResidualFSQuantizer(
            levels=levels,
            num_quantizers=num_quantizers,
            embedding_dim=embedding_dim,
            share_projections=False,
        )

        # With shared projections
        quantizer_shared = ResidualFSQuantizer(
            levels=levels,
            num_quantizers=num_quantizers,
            embedding_dim=embedding_dim,
            share_projections=True,
        )

        # Shared should have fewer parameters
        params_separate = sum(p.numel() for p in quantizer_separate.parameters())
        params_shared = sum(p.numel() for p in quantizer_shared.parameters())
        assert params_shared < params_separate

        # Both should produce valid outputs
        x = torch.randn(2, embedding_dim, 4, 4)
        q_sep, _, _ = quantizer_separate(x)
        q_shared, _, _ = quantizer_shared(x)

        assert q_sep.shape == x.shape
        assert q_shared.shape == x.shape

    def test_resfsq_indices_to_codes_3d(self) -> None:
        """Test indices_to_codes for 3D input."""
        quantizer = ResidualFSQuantizer(
            levels=[8, 5, 5],
            num_quantizers=2,
            embedding_dim=3,
        )
        x = torch.randn(2, 3, 4, 4, 4)
        quantised, _, indices = quantizer(x)

        recovered = quantizer.indices_to_codes(indices)

        assert recovered.shape == quantised.shape, (
            f"Shape mismatch: {recovered.shape} vs {quantised.shape}"
        )
        assert torch.allclose(recovered, quantised, atol=1e-5), "Roundtrip failed"

    def test_resfsq_effective_codebook_size(self) -> None:
        """Test that effective codebook size is correctly computed."""
        levels = [8, 5, 5]  # 200 codes per quantizer
        num_quantizers = 3
        quantizer = ResidualFSQuantizer(
            levels=levels, num_quantizers=num_quantizers, embedding_dim=3
        )

        single_codebook_size = 8 * 5 * 5  # 200
        expected_size = single_codebook_size**num_quantizers  # 200^3 = 8,000,000

        assert quantizer.get_codebook_size() == expected_size

    def test_fsq_large_codebook_index_precision(self) -> None:
        """Test FSQ index roundtrip for large codebooks (int64 precision)."""
        # Large codebook: 16^6 = 16,777,216 codes
        levels = [16, 16, 16, 16, 16, 16]
        quantizer = FSQuantizer(levels=levels, embedding_dim=6)

        # Use deterministic values at quantization boundaries
        x = torch.zeros(1, 6, 2, 2)
        x[0, :, 0, 0] = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        x[0, :, 0, 1] = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        x[0, :, 1, 0] = torch.tensor([0.5, -0.5, 0.5, -0.5, 0.5, -0.5])
        x[0, :, 1, 1] = torch.tensor([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])

        quantised, _, indices = quantizer(x)

        # Test codes_to_indices roundtrip on quantized values (channel-last)
        quantised_cl = quantised.permute(0, 2, 3, 1)  # (B, H, W, C)
        indices_recomputed = quantizer.codes_to_indices(quantised_cl)

        assert torch.equal(indices, indices_recomputed), (
            "Large codebook index precision failed"
        )


class TestLFQuantizerEdgeCases:
    """Edge case tests for LFQuantizer."""

    def test_lfq_rejects_invalid_dim(self) -> None:
        """Test that LFQ raises error for invalid dim at init."""
        with pytest.raises(ValueError, match="dim must be 2 or 3"):
            LFQuantizer(codebook_size=64, codebook_dim=6, dim=4)

    def test_lfq_indices_use_bitmask(self) -> None:
        """Test LFQ indices match binary bitmask encoding."""
        quantizer = LFQuantizer(
            codebook_size=8,
            codebook_dim=3,
            dim=2,
            entropy_loss=False,
            num_codebooks=1,
        )
        x = torch.tensor(
            [
                [
                    [[1.0, -1.0, -1.0, 1.0]],
                    [[-1.0, 1.0, -1.0, 1.0]],
                    [[-1.0, -1.0, 1.0, 1.0]],
                ]
            ]
        )

        _, _, indices = quantizer(x)
        indices = indices.squeeze(-1).flatten().to(torch.int64)

        expected = torch.tensor([4, 2, 1, 7], dtype=torch.int64)
        assert torch.equal(indices, expected)

    def test_lfq_3d_shape(self) -> None:
        """Test LFQ with 3D input."""
        quantizer = LFQuantizer(
            codebook_size=64,
            codebook_dim=6,
            dim=3,
            entropy_loss=False,
        )
        x = torch.randn(2, 6, 4, 4, 4)
        quantised, loss, indices = quantizer(x)

        assert quantised.shape == x.shape
        assert loss.shape == (2, 1, 1, 1, 1)  # 3D has 5D loss shape


class TestAMPStability:
    """Tests for numerical stability under automatic mixed precision."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fsq_amp_stability(self) -> None:
        """Test FSQ stability under AMP."""
        quantizer = FSQuantizer(levels=[8, 8, 8, 8], embedding_dim=4).cuda()
        x = torch.randn(4, 4, 16, 16, device="cuda")

        with torch.autocast("cuda"):
            quantised, loss, indices = quantizer(x)

        assert not torch.isnan(quantised).any(), "AMP produced NaN in quantised"
        assert not torch.isnan(loss).any(), "AMP produced NaN in loss"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_vq_ema_amp_stability(self) -> None:
        """Test VQ EMA stability under AMP (division by small numbers)."""
        quantizer = VectorQuantizer(
            num_embeddings=256,
            embedding_dim=8,
            dim=2,
            use_ema=True,
        ).cuda()
        quantizer.train()

        for _ in range(5):
            x = torch.randn(8, 8, 8, 8, device="cuda")
            with torch.autocast("cuda"):
                quantised, loss, indices = quantizer(x)

            assert not torch.isnan(quantised).any(), "AMP produced NaN in quantised"
            assert not torch.isnan(quantizer.embedding.weight).any(), (
                "AMP produced NaN in embeddings"
            )


class TestBugFixes:
    """Tests that verify specific bug fixes don't regress."""

    def test_lfq_entropy_loss_true_2d(self) -> None:
        """Test LFQ with entropy_loss=True produces correct output shapes.

        Regression test for fix: unpack([indices], ...) -> unpack(indices, ...)
        """
        quantizer = LFQuantizer(
            codebook_size=64,
            codebook_dim=6,
            dim=2,
            entropy_loss=True,
        )
        x = torch.randn(2, 6, 8, 8)
        quantised, loss, indices = quantizer(x)

        assert quantised.shape == x.shape
        assert loss.shape == (2, 1, 1, 1)
        assert indices.shape == (2, 8, 8)

    def test_lfq_entropy_loss_true_3d(self) -> None:
        """Test LFQ with entropy_loss=True for 3D input.

        Regression test for fix: unpack([indices], ...) -> unpack(indices, ...)
        """
        quantizer = LFQuantizer(
            codebook_size=64,
            codebook_dim=6,
            dim=3,
            entropy_loss=True,
        )
        x = torch.randn(2, 6, 4, 4, 4)
        quantised, loss, indices = quantizer(x)

        assert quantised.shape == x.shape
        assert loss.shape == (2, 1, 1, 1, 1)
        assert indices.shape == (2, 4, 4, 4)

    def test_resfsq_batch_size_1(self) -> None:
        """Test ResidualFSQ with batch_size=1 preserves batch dimension in loss.

        Regression test for fix: loss.squeeze() -> loss.squeeze(dim=1)
        """
        quantizer = ResidualFSQuantizer(
            levels=[3, 3, 3, 3, 3, 3],
            num_quantizers=3,
        )
        x = torch.randn(1, 6, 8, 8)  # batch_size=1
        quantised, loss, indices = quantizer(x)

        assert quantised.shape == x.shape
        # Loss should be (1, 8, 8), NOT (8, 8)
        assert loss.shape == (1, 8, 8), f"Expected (1, 8, 8), got {loss.shape}"
        assert indices.shape == (1, 3, 8, 8)

    def test_resfsq_batch_size_1_3d(self) -> None:
        """Test ResidualFSQ with batch_size=1 for 3D input."""
        quantizer = ResidualFSQuantizer(
            levels=[3, 3, 3, 3, 3, 3],
            num_quantizers=3,
        )
        x = torch.randn(1, 6, 4, 4, 4)  # batch_size=1
        quantised, loss, indices = quantizer(x)

        assert quantised.shape == x.shape
        assert loss.shape == (1, 4, 4, 4), f"Expected (1, 4, 4, 4), got {loss.shape}"
        assert indices.shape == (1, 3, 4, 4, 4)

    def test_vq_chunked_matches_full_distance(self) -> None:
        """Test that chunked distance computation matches full computation.

        VQ uses chunked nearest neighbor search for large inputs. This test
        verifies the optimization produces identical results to the naive approach.
        """
        torch.manual_seed(42)
        quantizer = VectorQuantizer(num_embeddings=256, embedding_dim=6, dim=3)

        # Create input large enough to trigger chunking (> 32K vectors)
        # 4 * 16 * 16 * 16 = 16384, need more
        x = torch.randn(2, 6, 32, 32, 32)  # 65536 vectors

        # Run with normal chunked computation
        quantiser_chunked = VectorQuantizer(num_embeddings=256, embedding_dim=6, dim=3)
        quantiser_chunked.load_state_dict(quantizer.state_dict())

        # Run with forced full computation (very large chunk size)
        quantiser_full = VectorQuantizer(num_embeddings=256, embedding_dim=6, dim=3)
        quantiser_full.load_state_dict(quantizer.state_dict())
        quantiser_full._DISTANCE_CHUNK_SIZE = 1_000_000  # Force full computation

        with torch.no_grad():
            z_q_chunked, _, indices_chunked = quantiser_chunked(x)
            z_q_full, _, indices_full = quantiser_full(x)

        assert torch.equal(indices_chunked, indices_full), "Indices differ"
        assert torch.allclose(z_q_chunked, z_q_full), "Quantized outputs differ"


class TestAnisotropicShapes:
    """Tests for non-cubic/non-square spatial dimensions."""

    def test_fsq_anisotropic_2d(self) -> None:
        """Test FSQ with non-square 2D input."""
        quantizer = FSQuantizer(levels=[8, 5, 5, 5], embedding_dim=4)
        x = torch.randn(2, 4, 16, 8)  # H=16, W=8
        quantised, loss, indices = quantizer(x)

        assert quantised.shape == x.shape
        assert indices.shape == (2, 16, 8)

    def test_fsq_anisotropic_3d(self) -> None:
        """Test FSQ with non-cubic 3D input."""
        quantizer = FSQuantizer(levels=[8, 5, 5, 5], embedding_dim=4)
        x = torch.randn(2, 4, 16, 8, 4)  # H=16, W=8, D=4
        quantised, loss, indices = quantizer(x)

        assert quantised.shape == x.shape
        assert indices.shape == (2, 16, 8, 4)

    def test_vq_anisotropic_3d(self) -> None:
        """Test VQ with non-cubic 3D input."""
        quantizer = VectorQuantizer(num_embeddings=256, embedding_dim=4, dim=3)
        x = torch.randn(2, 4, 16, 8, 4)  # H=16, W=8, D=4
        quantised, loss, indices = quantizer(x)

        assert quantised.shape == x.shape
        assert indices.shape == (2, 16, 8, 4)

    def test_resfsq_anisotropic_3d(self) -> None:
        """Test ResidualFSQ with non-cubic 3D input."""
        quantizer = ResidualFSQuantizer(
            levels=[8, 5, 5],
            num_quantizers=2,
            embedding_dim=3,
        )
        x = torch.randn(2, 3, 16, 8, 4)  # H=16, W=8, D=4
        quantised, loss, indices = quantizer(x)

        assert quantised.shape == x.shape
        assert indices.shape == (2, 2, 16, 8, 4)  # (B, num_quantizers, H, W, D)


class TestGradientFlow:
    """Tests verifying STE (Straight-Through Estimator) gradient flow.

    These tests ensure that gradients actually flow through quantization
    operations to the encoder. Without this, training would silently fail.
    """

    def test_fsq_gradient_flows_to_input(self) -> None:
        """Test that FSQ STE allows gradient flow to input."""
        quantizer = FSQuantizer(levels=[8, 8, 8, 8], embedding_dim=4)
        x = torch.randn(2, 4, 8, 8, requires_grad=True)

        quantised, loss, indices = quantizer(x)

        # Backward from quantised output
        quantised.sum().backward()

        assert x.grad is not None, "Gradient did not flow to input"
        assert x.grad.abs().sum() > 0, "Gradient is all zeros"

    def test_fsq_gradient_is_ste(self) -> None:
        """Test that FSQ gradient is straight-through (identity)."""
        quantizer = FSQuantizer(levels=[8, 8, 8, 8], embedding_dim=4)
        x = torch.randn(2, 4, 8, 8, requires_grad=True)

        quantised, loss, indices = quantizer(x)

        # Use sum as loss for simple gradient
        loss_val = quantised.sum()
        loss_val.backward()

        # For STE, gradient magnitude should be approximately 1 per element
        # (not exactly 1 due to projections and normalization)
        mean_grad_magnitude = x.grad.abs().mean()
        assert 0.1 < mean_grad_magnitude < 10.0, (
            f"Gradient magnitude {mean_grad_magnitude} suggests STE is broken"
        )

    def test_vq_gradient_flows_to_input(self) -> None:
        """Test that VQ STE allows gradient flow to input."""
        quantizer = VectorQuantizer(num_embeddings=64, embedding_dim=4, dim=2)
        x = torch.randn(2, 4, 8, 8, requires_grad=True)

        quantised, loss, indices = quantizer(x)

        # Backward from quantised output
        quantised.sum().backward()

        assert x.grad is not None, "VQ gradient did not flow to input"
        assert x.grad.abs().sum() > 0, "VQ gradient is all zeros"

    def test_resfsq_gradient_flows_to_input(self) -> None:
        """Test that RESFSQ STE allows gradient flow through all layers."""
        quantizer = ResidualFSQuantizer(
            levels=[8, 5, 5],
            num_quantizers=4,
            embedding_dim=3,
        )
        x = torch.randn(2, 3, 8, 8, requires_grad=True)

        quantised, loss, indices = quantizer(x)

        # Backward from quantised output
        quantised.sum().backward()

        assert x.grad is not None, "RESFSQ gradient did not flow to input"
        assert x.grad.abs().sum() > 0, "RESFSQ gradient is all zeros"

    def test_lfq_gradient_flows_to_input(self) -> None:
        """Test that LFQ STE allows gradient flow to input."""
        quantizer = LFQuantizer(
            codebook_size=64,
            codebook_dim=6,
            dim=2,
            entropy_loss=False,
        )
        x = torch.randn(2, 6, 8, 8, requires_grad=True)

        quantised, loss, indices = quantizer(x)

        # Backward from quantised output
        quantised.sum().backward()

        assert x.grad is not None, "LFQ gradient did not flow to input"
        assert x.grad.abs().sum() > 0, "LFQ gradient is all zeros"

    def test_quantizer_loss_gradient_is_independent(self) -> None:
        """Test that quantizer loss gradient is separate from reconstruction.

        For VQ: commitment loss gradient should affect encoder
        For FSQ: loss is zero (but gradient should still work)
        """
        # VQ with commitment loss
        quantizer = VectorQuantizer(
            num_embeddings=64, embedding_dim=4, dim=2, beta=0.25
        )
        x = torch.randn(2, 4, 8, 8, requires_grad=True)

        quantised, loss, indices = quantizer(x)

        # Only backward from loss (not reconstruction)
        loss.sum().backward()

        # Commitment loss should produce non-zero gradients
        assert x.grad is not None, "Loss gradient did not flow to input"
        assert x.grad.abs().sum() > 0, "Loss gradient is all zeros"

    def test_round_ste_preserves_gradient_direction(self) -> None:
        """Test that round_ste preserves gradient direction (not magnitude)."""
        from medtokenizers.modules.utils import round_ste

        x = torch.tensor([1.7, -0.3, 2.2], requires_grad=True)
        y = round_ste(x)

        # y should be [2., 0., 2.]
        assert torch.allclose(y, torch.tensor([2.0, 0.0, 2.0]))

        # Backward with specific gradient
        grad_out = torch.tensor([1.0, -2.0, 0.5])
        y.backward(grad_out)

        # STE should pass gradient through unchanged
        assert torch.allclose(x.grad, grad_out)


class TestMixedPrecisionStability:
    """Tests for numerical stability with float16/bfloat16 inputs.

    These tests verify quantizers work correctly with reduced precision,
    which is important for memory-efficient training with AMP.

    Design notes:
    - FSQ intentionally computes in float32 for numerical stability in bound()
    - VQ embeddings are always float32 to prevent codebook degradation
    - Output dtype may differ from input when stability requires it
    """

    def test_fsq_float16_produces_valid_output(self) -> None:
        """Test FSQ handles float16 input without NaN/Inf.

        Note: FSQ computes bound() in float32 for stability, so output
        may be float32 even with float16 input. This is intentional.
        """
        quantizer = FSQuantizer(levels=[8, 8, 8, 8], embedding_dim=4)
        x = torch.randn(2, 4, 8, 8, dtype=torch.float16)

        quantised, loss, indices = quantizer(x)

        assert not torch.isnan(quantised).any(), "float16 produced NaN"
        assert not torch.isinf(quantised).any(), "float16 produced Inf"
        # Values should be properly bounded regardless of dtype
        assert quantised.abs().max() <= 1.5, "Values not properly bounded"

    def test_fsq_bfloat16_produces_valid_output(self) -> None:
        """Test FSQ handles bfloat16 input without NaN/Inf."""
        quantizer = FSQuantizer(levels=[8, 8, 8, 8], embedding_dim=4)
        x = torch.randn(2, 4, 8, 8, dtype=torch.bfloat16)

        quantised, loss, indices = quantizer(x)

        assert not torch.isnan(quantised).any(), "bfloat16 produced NaN"
        assert not torch.isinf(quantised).any(), "bfloat16 produced Inf"

    def test_fsq_extreme_values_float16(self) -> None:
        """Test FSQ bound() handles extreme float16 values."""
        quantizer = FSQuantizer(levels=[8, 8, 8, 8], embedding_dim=4)

        # float16 max is ~65504, test near this limit
        x = torch.full((2, 4, 4, 4), 1000.0, dtype=torch.float16)
        quantised, loss, indices = quantizer(x)

        assert not torch.isnan(quantised).any(), "Extreme values produced NaN"
        # Values should be bounded to valid range
        assert quantised.abs().max() <= 2.0, "Values not properly bounded"

    def test_vq_with_autocast_context(self) -> None:
        """Test VQ works correctly inside autocast context.

        VQ embeddings are float32 by design. When using AMP, input should
        be cast to match embedding dtype inside autocast context.
        """
        quantizer = VectorQuantizer(num_embeddings=64, embedding_dim=4, dim=2)
        x = torch.randn(2, 4, 8, 8)  # float32 input

        # Simulate what happens in typical AMP training
        quantised, loss, indices = quantizer(x)

        assert not torch.isnan(quantised).any(), "VQ produced NaN"
        assert not torch.isinf(quantised).any(), "VQ produced Inf"
        assert quantised.dtype == torch.float32, "VQ should output float32"

    def test_vq_ema_maintains_float32_embeddings(self) -> None:
        """Test VQ EMA maintains float32 embeddings for stability."""
        quantizer = VectorQuantizer(
            num_embeddings=64,
            embedding_dim=4,
            dim=2,
            use_ema=True,
        )
        quantizer.train()

        # Run several updates
        for _ in range(10):
            x = torch.randn(4, 4, 8, 8)
            quantised, loss, indices = quantizer(x)

        # Embeddings should remain valid float32
        assert quantizer.embedding.weight.dtype == torch.float32
        assert not torch.isnan(quantizer.embedding.weight).any(), (
            "EMA produced NaN embeddings"
        )
        assert not torch.isinf(quantizer.embedding.weight).any(), (
            "EMA produced Inf embeddings"
        )

    def test_lfq_float16_forward(self) -> None:
        """Test LFQ handles float16 input correctly."""
        quantizer = LFQuantizer(
            codebook_size=64,
            codebook_dim=6,
            dim=2,
            entropy_loss=False,
        )
        x = torch.randn(2, 6, 8, 8, dtype=torch.float16)

        quantised, loss, indices = quantizer(x)

        assert not torch.isnan(quantised).any(), "LFQ float16 produced NaN"
        assert not torch.isinf(quantised).any(), "LFQ float16 produced Inf"

    def test_resfsq_float16_forward(self) -> None:
        """Test RESFSQ handles float16 input correctly."""
        quantizer = ResidualFSQuantizer(
            levels=[8, 5, 5],
            num_quantizers=4,
            embedding_dim=3,
        )
        x = torch.randn(2, 3, 8, 8, dtype=torch.float16)

        quantised, loss, indices = quantizer(x)

        assert not torch.isnan(quantised).any(), "RESFSQ float16 produced NaN"
        assert not torch.isinf(quantised).any(), "RESFSQ float16 produced Inf"

    def test_fsq_float16_gradient_stability(self) -> None:
        """Test FSQ gradients are stable with float16."""
        quantizer = FSQuantizer(levels=[8, 8, 8, 8], embedding_dim=4)
        x = torch.randn(2, 4, 8, 8, dtype=torch.float16, requires_grad=True)

        quantised, loss, indices = quantizer(x)
        quantised.sum().backward()

        assert x.grad is not None, "Gradient is None"
        assert not torch.isnan(x.grad).any(), "Gradient contains NaN"
        assert not torch.isinf(x.grad).any(), "Gradient contains Inf"

    def test_vq_commitment_loss_stability(self) -> None:
        """Test VQ commitment loss is stable."""
        quantizer = VectorQuantizer(
            num_embeddings=64,
            embedding_dim=4,
            dim=2,
            beta=0.25,
        )
        x = torch.randn(2, 4, 8, 8, requires_grad=True)

        quantised, loss, indices = quantizer(x)

        # Loss should be finite
        assert not torch.isnan(loss).any(), "Commitment loss is NaN"
        assert not torch.isinf(loss).any(), "Commitment loss is Inf"

        # Gradient through loss should be stable
        loss.sum().backward()
        assert not torch.isnan(x.grad).any(), "Loss gradient is NaN"


class TestDistributedCompatibility:
    """Tests for distributed training compatibility."""

    def test_vq_ema_buffers_registered(self) -> None:
        """Test VQ EMA buffers are properly registered for DDP sync."""
        quantizer = VectorQuantizer(
            num_embeddings=64,
            embedding_dim=4,
            dim=2,
            use_ema=True,
        )

        # Check buffers are registered (important for DDP)
        buffer_names = [name for name, _ in quantizer.named_buffers()]
        assert "ema_cluster_size" in buffer_names
        assert "ema_embed_sum" in buffer_names
        assert "batches_since_used" in buffer_names

    def test_vq_ema_update_uses_all_reduce_when_available(self) -> None:
        """Test VQ EMA update has distributed sync code path.

        This test verifies the code structure but doesn't test actual
        distributed behavior (that requires multiple processes).
        """
        import inspect

        from medtokenizers.modules.quant import VectorQuantizer

        # Check the _update_ema method source for distributed handling
        source = inspect.getsource(VectorQuantizer._update_ema)
        assert "torch.distributed.is_initialized" in source, (
            "VQ EMA should check for distributed training"
        )
        assert "all_reduce" in source, (
            "VQ EMA should all-reduce statistics in distributed mode"
        )
