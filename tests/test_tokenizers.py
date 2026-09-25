import pytest
import torch

from medtokenizers.networks import ContinuousTokenizer, DiscreteTokenizer


@pytest.fixture
def base_kwargs() -> dict:
    return {
        "in_channels": 1,
        "out_channels": 1,
        "channels": 32,
        "channels_mult": [1, 2],
        "num_res_blocks": 1,
        "attn_resolutions": [8],
        "dropout": 0.0,
        "resolution": 16,
        "spatial_compression": 4,
    }


def test_continuous_tokenizer_autoencoder_2d(base_kwargs: dict) -> None:
    model = ContinuousTokenizer(
        dim=2,
        z_channels=4,
        embedding_dim=4,
        z_factor=1,
        latent_channels=4,
        formulation="AE",
        **base_kwargs,
    ).to(torch.float32)
    x = torch.randn(2, 1, 16, 16, dtype=torch.float32)

    result = model(x)

    assert result["reconstructions"].shape == x.shape
    assert result["posteriors"][0].shape[0] == x.shape[0]


def test_continuous_tokenizer_variational_3d(base_kwargs: dict) -> None:
    model = ContinuousTokenizer(
        dim=3,
        z_channels=4,
        embedding_dim=4,
        z_factor=2,
        latent_channels=4,
        formulation="VAE",
        **base_kwargs,
    ).to(torch.float32)
    x = torch.randn(1, 1, 16, 16, 16, dtype=torch.float32)

    result = model(x)
    mean, logvar = result["posteriors"]

    assert result["reconstructions"].shape == x.shape
    assert mean.shape == logvar.shape
    assert torch.all(logvar <= model.distribution.max_logvar)


def test_discrete_tokenizer_vector_quantizer_round_trip(base_kwargs: dict) -> None:
    model = DiscreteTokenizer(
        dim=2,
        z_channels=4,
        embedding_dim=4,
        quantizer="VQ",
        num_embeddings=16,
        levels=[2] * 4,
        num_quantizers=2,
        **base_kwargs,
    ).to(torch.float32)
    x = torch.randn(2, 1, 16, 16, dtype=torch.float32)

    encoded, quantised, loss = model.encode(x)
    decoded = model.decode(quantised)

    assert decoded.shape == x.shape
    assert loss.shape == (x.shape[0], 1, 1, 1)
    assert encoded.shape[0] == x.shape[0]


def test_discrete_tokenizer_residual_fsq_handles_flat_tokens(base_kwargs: dict) -> None:
    model = DiscreteTokenizer(
        dim=3,
        z_channels=4,
        embedding_dim=6,
        quantizer="RESFSQ",
        levels=[2] * 6,
        num_quantizers=3,
        **base_kwargs,
    ).to(torch.float32)
    model.eval()
    x = torch.randn(1, 1, 16, 16, 24, dtype=torch.float32)

    with torch.no_grad():
        tokens = model.tokenize(x)
        spatial_shape = tokens.shape[2:]
        recon = model.detokenize(tokens.flatten(2), spatial_shape=spatial_shape)

        assert torch.allclose(recon, model.detokenize(tokens), atol=1e-5)
    with pytest.raises(ValueError, match="spatial_shape"):
        model.detokenize(tokens.flatten(2))


def test_discrete_tokenizer_rejects_unknown_quantizer(base_kwargs: dict) -> None:
    with pytest.raises(ValueError):
        DiscreteTokenizer(
            dim=2,
            z_channels=4,
            embedding_dim=4,
            quantizer="UNKNOWN",
            levels=[2] * 4,
            num_quantizers=1,
            **base_kwargs,
        )


class TestDeterminism:
    """Tests verifying deterministic behavior in eval mode.

    Tokenizers should produce identical outputs for identical inputs
    when in eval mode. This is critical for reproducible inference.
    """

    def test_continuous_ae_eval_deterministic(self, base_kwargs: dict) -> None:
        """Test that AE produces identical outputs in eval mode."""
        model = ContinuousTokenizer(
            dim=2,
            z_channels=4,
            latent_channels=4,
            formulation="AE",
            **base_kwargs,
        )
        model.eval()

        x = torch.randn(2, 1, 16, 16)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.allclose(output1.reconstructions, output2.reconstructions), (
            "AE output not deterministic in eval mode"
        )

    def test_continuous_vae_eval_stochastic(self, base_kwargs: dict) -> None:
        """Test that VAE sampling is stochastic (different each call).

        Note: This is expected behavior - VAE samples z ~ N(mu, sigma).
        Use the mean for deterministic inference.
        """
        model = ContinuousTokenizer(
            dim=2,
            z_channels=4,
            latent_channels=4,
            formulation="VAE",
            separate_quant_conv=False,
            **base_kwargs,
        )
        model.eval()

        x = torch.randn(2, 1, 16, 16)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        # VAE with combined quant_conv should be stochastic (different outputs)
        # If they're identical, sampling may be broken
        assert not torch.allclose(output1.reconstructions, output2.reconstructions), (
            "VAE output is unexpectedly deterministic - sampling may be broken"
        )

    def test_discrete_fsq_eval_deterministic(self, base_kwargs: dict) -> None:
        """Test that discrete FSQ tokenizer is deterministic in eval mode."""
        model = DiscreteTokenizer(
            dim=2,
            z_channels=4,
            embedding_dim=4,
            quantizer="FSQ",
            levels=[8, 5, 5, 5],
            **base_kwargs,
        )
        model.eval()

        x = torch.randn(2, 1, 16, 16)

        with torch.no_grad():
            indices1, quant1, _ = model.encode(x)
            indices2, quant2, _ = model.encode(x)

        assert torch.equal(indices1, indices2), "FSQ indices not deterministic"
        assert torch.allclose(quant1, quant2), "FSQ quantized output not deterministic"

    def test_discrete_vq_eval_deterministic(self, base_kwargs: dict) -> None:
        """Test that discrete VQ tokenizer is deterministic in eval mode."""
        model = DiscreteTokenizer(
            dim=2,
            z_channels=4,
            embedding_dim=4,
            quantizer="VQ",
            num_embeddings=64,
            **base_kwargs,
        )
        model.eval()

        x = torch.randn(2, 1, 16, 16)

        with torch.no_grad():
            indices1, quant1, _ = model.encode(x)
            indices2, quant2, _ = model.encode(x)

        assert torch.equal(indices1, indices2), "VQ indices not deterministic"
        assert torch.allclose(quant1, quant2), "VQ quantized output not deterministic"

    def test_tokenize_detokenize_consistent(self, base_kwargs: dict) -> None:
        """Test tokenize/detokenize roundtrip is consistent across calls."""
        model = DiscreteTokenizer(
            dim=2,
            z_channels=4,
            embedding_dim=4,
            quantizer="FSQ",
            levels=[8, 5, 5, 5],
            **base_kwargs,
        )
        model.eval()

        x = torch.randn(2, 1, 16, 16)

        with torch.no_grad():
            tokens = model.tokenize(x)
            recon1 = model.detokenize(tokens)
            recon2 = model.detokenize(tokens)

        assert torch.allclose(recon1, recon2), "Detokenize not deterministic"

    @pytest.mark.parametrize(
        "quantizer_kwargs",
        [
            {"quantizer": "VQ", "num_embeddings": 64, "use_norm": False},
            {"quantizer": "VQ", "num_embeddings": 64, "use_norm": True},
            {"quantizer": "FSQ", "levels": [8, 5, 5, 5]},
            {"quantizer": "LFQ", "codebook_size": 64, "codebook_dim": 6},
            {"quantizer": "RESFSQ", "levels": [8, 8, 8, 8], "num_codebooks": 2},
        ],
        ids=["vq", "vq-norm", "fsq", "lfq", "resfsq"],
    )
    def test_detokenize_matches_forward(
        self, base_kwargs: dict, quantizer_kwargs: dict
    ) -> None:
        """Test that detokenize(tokenize(x)) equals the forward reconstruction."""
        model = DiscreteTokenizer(
            dim=2, z_channels=4, embedding_dim=4, **quantizer_kwargs, **base_kwargs
        )
        if quantizer_kwargs["quantizer"] == "VQ":
            with torch.no_grad():
                model.quantizer.embedding.weight.normal_(std=3.0)
        model.eval()

        # A 4 x 4 latent with embedding_dim=4 makes channels-first and channels-last codes the same shape.
        x = torch.randn(2, 1, 16, 16)

        with torch.no_grad():
            recon = model(x).reconstructions
            tokens = model.tokenize(x)
            roundtrip = model.detokenize(tokens)
            spatial_axis = 2 if quantizer_kwargs["quantizer"] == "RESFSQ" else 1
            flat = model.detokenize(
                tokens.flatten(spatial_axis, spatial_axis + 1), spatial_shape=(4, 4)
            )

        assert torch.allclose(roundtrip, recon, atol=1e-5)
        assert torch.allclose(flat, recon, atol=1e-5)

    def test_seeded_vae_reproducible(self, base_kwargs: dict) -> None:
        """Test that VAE is reproducible with manual seeding."""
        model = ContinuousTokenizer(
            dim=2,
            z_channels=4,
            latent_channels=4,
            formulation="VAE",
            **base_kwargs,
        )
        model.eval()

        x = torch.randn(2, 1, 16, 16)

        # Same seed should give same output
        torch.manual_seed(42)
        with torch.no_grad():
            output1 = model(x)

        torch.manual_seed(42)
        with torch.no_grad():
            output2 = model(x)

        assert torch.allclose(output1.reconstructions, output2.reconstructions), (
            "VAE not reproducible with same seed"
        )
