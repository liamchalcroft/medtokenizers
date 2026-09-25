"""Discrete latent space tokenizers (VQ-VAE, FSQ, RESFSQ, LFQ).

This module implements tokenizers that map images to discrete latent codes
from a finite vocabulary. These are ideal for autoregressive models and
language model-based generation.

Discrete vs Continuous Tokenizers
---------------------------------
Discrete tokenizers (this module):
- Produce integer codes from finite vocabulary K
- Output: indices ∈ {0, 1, ..., K-1}^(H'×W'×D')
- Suitable for: transformers, autoregressive models, LLMs

Continuous tokenizers (see continuous.py):
- Produce real-valued vectors z ∈ ℝ^d
- Suitable for: diffusion models, VAE-based generation

Available Quantization Methods
------------------------------
1. **VQ (Vector Quantization)**: Classic learned codebook
   - K learnable embedding vectors
   - Commitment loss for encoder, codebook loss for embeddings
   - Requires careful initialization to avoid collapse

2. **FSQ (Finite Scalar Quantization)**: Implicit codebook
   - Codebook defined implicitly by quantization levels
   - No learning of codebook entries
   - No collapse issues, simpler training

3. **RESFSQ (Residual FSQ)**: Hierarchical FSQ
   - Multiple FSQ layers encoding residuals
   - Exponentially larger effective codebook
   - Good balance of simplicity and expressiveness

4. **LFQ (Lookup-Free Quantization)**: Binary codes
   - Codebook is all binary vectors {-1, +1}^d
   - Optional entropy regularization
   - Extreme simplicity, good for very large vocabularies

Training Objective
------------------
```
L = L_reconstruction + λ * L_quantization
```

Where L_quantization depends on method:
- VQ: commitment + codebook loss
- FSQ/RESFSQ: none (implicit codebook)
- LFQ: optional entropy regularization

Architecture
------------
```
Input -> Encoder -> quant_conv -> Quantizer -> Decoder -> Reconstruction
                                    |
                                    v
                               Indices (discrete codes)
```
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Optional

import torch
import torch.nn as nn

from medtokenizers.modules.base import BaseTokenizer
from medtokenizers.modules.layers import Decoder, Encoder
from medtokenizers.modules.quant import (
    FSQuantizer,
    LFQuantizer,
    ResidualFSQuantizer,
    VectorQuantizer,
)
from medtokenizers.modules.utils import validate_tensor_input
from medtokenizers.networks._types import NetworkEval

if TYPE_CHECKING:
    from jaxtyping import Float, Int


logger = logging.getLogger(__name__)


QuantizerType = Literal["VQ", "FSQ", "LFQ", "RESFSQ"]


class DiscreteTokenizer(BaseTokenizer):
    """Discrete latent tokenizer for medical imaging.

    This tokenizer learns discrete latent representations using various
    quantization methods, enabling the use of language models and
    autoregressive architectures for medical image generation.

    The Key Insight
    ---------------
    By quantizing continuous encoder outputs to a finite vocabulary,
    we convert the image generation problem into a sequence modeling
    problem that can leverage powerful transformer architectures.

    Quantization Methods
    --------------------
    Choose based on your use case:

    - **VQ**: Maximum expressiveness, but requires careful training
      to avoid codebook collapse. Best for small codebooks (~1K).

    - **FSQ**: Stable training with implicit codebook. No collapse.
      Good default choice for most applications.

    - **RESFSQ**: Massive effective codebook via residual stacking.
      Use when you need very high fidelity reconstruction.

    - **LFQ**: Binary codes for extreme simplicity. Good for
      very large-scale generation with lightweight decoders.

    Architecture Details
    --------------------
    ::

        Encoder Path:
        Input -> Conv_in -> ResBlocks -> Downsample -> Conv_out -> z_continuous

        Quantization:
        z_continuous -> quant_conv -> Quantizer -> (indices, z_quantized)

        Decoder Path:
        z_quantized -> post_quant_conv -> ResBlocks -> Upsample -> Output

    Memory Optimization
    -------------------
    For 3D volumes, the model automatically:
    - Uses channels_last_3d memory format
    - Supports gradient checkpointing

    Args:
        dim: Spatial dimensionality (2 for 2D, 3 for 3D)
        in_channels: Number of input channels
        out_channels: Number of output channels
        z_channels: Encoder output channels (before quant_conv)
        embedding_dim: Dimension of quantized embeddings
        channels: Base channel count for encoder/decoder
        channels_mult: Channel multipliers per resolution
        num_res_blocks: Residual blocks per resolution
        attn_resolutions: Resolutions for self-attention
        dropout: Dropout probability
        resolution: Input spatial resolution
        spatial_compression: Total downsampling factor
        quantizer: Quantization method ("VQ", "FSQ", "LFQ", "RESFSQ")
        num_embeddings: Codebook size for VQ (default: 1024)
        beta: Commitment loss weight for VQ (default: 0.25)
        use_norm: Normalize VQ embeddings (cosine similarity)
        levels: FSQ quantization levels (e.g., [8, 5, 5, 5])
        num_codebooks: Number of quantizers for RESFSQ/LFQ
        codebook_size: LFQ codebook size (must be power of 2)
        codebook_dim: LFQ code dimension
        entropy_loss_weight: LFQ entropy regularization weight
        commitment_loss_weight: LFQ commitment loss weight
        quant_temp: Temperature for soft quantization
        name: Model identifier
        **kwargs: Additional encoder/decoder arguments

    Example:
        >>> # FSQ tokenizer for 3D medical volumes
        >>> model = DiscreteTokenizer(
        ...     dim=3,
        ...     in_channels=1,
        ...     out_channels=1,
        ...     z_channels=128,
        ...     embedding_dim=6,
        ...     quantizer='FSQ',
        ...     levels=[8, 5, 5, 5],  # 1000 codes
        ...     spatial_compression=8,
        ... )
        >>>
        >>> # Tokenize to discrete codes
        >>> volume = torch.randn(1, 1, 128, 128, 128)
        >>> with model.inference_mode():
        ...     indices = model.tokenize(volume)  # (1, 16, 16, 16)
        ...     reconstructed = model.detokenize(indices)
        >>>
        >>> # Training forward pass
        >>> output = model(volume)
        >>> recon_loss = F.l1_loss(output['reconstructions'], volume)
        >>> quant_loss = output['quant_loss'].mean()
        >>> total_loss = recon_loss + quant_loss

    References:
        van den Oord et al. "Neural Discrete Representation Learning" (VQ-VAE)
        Mentzer et al. "Finite Scalar Quantization: VQ-VAE Made Simple" (FSQ)
        Yu et al. "Language Model Beats Diffusion" (LFQ in MagViT-2)
    """

    def __init__(
        self,
        dim: int,
        in_channels: int = 1,
        out_channels: int = 1,
        z_channels: int = 4,
        embedding_dim: int = 6,
        channels: int = 64,
        channels_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (),
        dropout: float = 0.0,
        resolution: int = 256,
        spatial_compression: int = 4,
        quantizer: QuantizerType = "RESFSQ",
        use_encoder_mid: bool = False,
        use_output_nonlinearity: bool = False,
        decoder_blocks_per_stage: Optional[list[int]] = None,
        # VQ specific
        num_embeddings: int = 1024,
        beta: float = 0.25,
        use_norm: bool = False,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        # FSQ specific
        levels: list[int] | None = None,
        # RESFSQ specific
        num_codebooks: int = 1,
        # LFQ specific
        codebook_size: Optional[int] = None,
        codebook_dim: Optional[int] = None,
        entropy_loss_weight: float = 0.1,
        commitment_loss_weight: float = 0.25,
        quant_temp: float = 0.01,
        name: str = "DiscreteTokenizer",
        **kwargs: Any,
    ) -> None:
        super().__init__(dim=dim, name=name)

        # Validate inputs
        if dim not in [2, 3]:
            raise ValueError(f"dim must be 2 or 3, got {dim}")
        if quantizer not in ["VQ", "FSQ", "LFQ", "RESFSQ"]:
            raise ValueError(f"quantizer must be VQ/FSQ/LFQ/RESFSQ, got {quantizer}")

        if decoder_blocks_per_stage is None:
            decoder_blocks_per_stage = [2, 2, 0]

        # Store config for serialization
        self.config = {
            "dim": dim,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "z_channels": z_channels,
            "embedding_dim": embedding_dim,
            "channels": channels,
            "channels_mult": list(channels_mult),
            "num_res_blocks": num_res_blocks,
            "attn_resolutions": list(attn_resolutions),
            "dropout": dropout,
            "resolution": resolution,
            "spatial_compression": spatial_compression,
            "quantizer": quantizer,
            "use_encoder_mid": use_encoder_mid,
            "use_output_nonlinearity": use_output_nonlinearity,
            "decoder_blocks_per_stage": decoder_blocks_per_stage,
            "num_embeddings": num_embeddings,
            "beta": beta,
            "use_norm": use_norm,
            "use_ema": use_ema,
            "ema_decay": ema_decay,
            "levels": levels if levels is not None else [8, 8, 8],
            "num_codebooks": num_codebooks,
            "codebook_size": codebook_size,
            "codebook_dim": codebook_dim,
            "entropy_loss_weight": entropy_loss_weight,
            "commitment_loss_weight": commitment_loss_weight,
            "quant_temp": quant_temp,
            "name": name,
        }
        self.config.update(kwargs)

        self.embedding_dim = embedding_dim
        self.spatial_compression = spatial_compression
        self.quantizer_type = quantizer

        # Prepare kwargs for encoder/decoder
        layer_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "channels": channels,
            "channels_mult": channels_mult,
            "num_res_blocks": num_res_blocks,
            "attn_resolutions": attn_resolutions,
            "dropout": dropout,
            "resolution": resolution,
            "spatial_compression": spatial_compression,
            "use_encoder_mid": use_encoder_mid,
            "use_output_nonlinearity": use_output_nonlinearity,
            "decoder_blocks_per_stage": decoder_blocks_per_stage,
        }
        layer_kwargs.update(kwargs)

        # Validate that spatial_compression divides resolution evenly
        # This prevents silent shape mismatches during forward pass
        if resolution % spatial_compression != 0:
            raise ValueError(
                f"Resolution {resolution}x{resolution} is not divisible by "
                f"spatial_compression={spatial_compression}. Expected latent size would be "
                f"({resolution // spatial_compression}x{resolution // spatial_compression}) but this causes "
                f"shape mismatch with target. Use compression that divides resolution evenly "
                f"(powers of 2: 1, 2, 4, 8, 16)."
            )

        # Build encoder and decoder
        self.encoder = Encoder(dim=dim, z_channels=z_channels, **layer_kwargs)
        self.decoder = Decoder(dim=dim, z_channels=z_channels, **layer_kwargs)

        # Latent projection layers (1x1 convs)
        conv_class = nn.Conv2d if dim == 2 else nn.Conv3d
        self.quant_conv = conv_class(z_channels, embedding_dim, kernel_size=1)
        self.post_quant_conv = conv_class(embedding_dim, z_channels, kernel_size=1)

        # Initialize quantizer based on type
        self.quantizer = self._build_quantizer(
            quantizer=quantizer,
            dim=dim,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            beta=beta,
            use_norm=use_norm,
            use_ema=use_ema,
            ema_decay=ema_decay,
            levels=levels,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            entropy_loss_weight=entropy_loss_weight,
            commitment_loss_weight=commitment_loss_weight,
            quant_temp=quant_temp,
        )

        # Log model info
        num_params = sum(p.numel() for p in self.parameters())
        logger.info(f"{self.name} based on {quantizer}-VAE")
        logger.info(f"Parameters: {num_params:,}")
        logger.info(f"z_channels={z_channels}, embedding_dim={embedding_dim}")

    def _build_quantizer(
        self,
        quantizer: str,
        dim: int,
        embedding_dim: int,
        num_embeddings: int,
        beta: float,
        use_norm: bool,
        use_ema: bool,
        ema_decay: float,
        levels: list[int] | None,
        num_codebooks: int,
        codebook_size: Optional[int],
        codebook_dim: Optional[int],
        entropy_loss_weight: float,
        commitment_loss_weight: float,
        quant_temp: float,
    ) -> nn.Module:
        """Build the appropriate quantizer module."""
        if quantizer == "VQ":
            return VectorQuantizer(
                dim=dim,
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                beta=beta,
                use_norm=use_norm,
                use_ema=use_ema,
                ema_decay=ema_decay,
                reset_unused_codes=use_ema,
            )
        elif quantizer == "FSQ":
            return FSQuantizer(
                embedding_dim=embedding_dim,
                levels=levels or [8, 5, 5, 5],
            )
        elif quantizer == "RESFSQ":
            return ResidualFSQuantizer(
                embedding_dim=embedding_dim,
                levels=levels or [8, 8, 8],
                num_quantizers=num_codebooks,
            )
        elif quantizer == "LFQ":
            if codebook_size is None or codebook_dim is None:
                raise ValueError("LFQ requires codebook_size and codebook_dim")
            return LFQuantizer(
                dim=dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                num_codebooks=num_codebooks,
                embedding_dim=embedding_dim,
                entropy_loss_weight=entropy_loss_weight,
                commitment_loss_weight=commitment_loss_weight,
                default_temp=quant_temp,
                entropy_loss=True,
            )
        else:
            raise ValueError(f"Unknown quantizer: {quantizer}")

    def to(self, *args, **kwargs) -> DiscreteTokenizer:
        """Move and/or cast the model, keeping the quantizer dtype in sync.

        The quantizer keeps its own ``dtype`` attribute (used by its numerical
        guards). It is updated *only* when a dtype is actually supplied, so a
        plain device move such as ``model.to("cuda")`` no longer silently resets
        it to ``float32``. A dtype may be passed either positionally
        (``model.to(torch.float16)``) or as the ``dtype`` keyword.

        Args:
            *args: Positional arguments forwarded to :meth:`torch.nn.Module.to`.
            **kwargs: Keyword arguments forwarded to :meth:`torch.nn.Module.to`.

        Returns:
            ``self``, after the move/cast has been applied.
        """
        dtype = kwargs.get("dtype")
        if dtype is None:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype = arg
                    break
        if dtype is not None:
            self.quantizer.dtype = dtype
        return super().to(*args, **kwargs)

    def encode(
        self, x: Float[torch.Tensor, "batch channels *spatial"]
    ) -> tuple[
        Int[torch.Tensor, "batch *spatial_indices"],
        Float[torch.Tensor, "batch embedding_dim *spatial_compressed"],
        Float[torch.Tensor, "..."],
    ]:
        """Encode input to discrete codes.

        Passes input through encoder, projects to embedding dimension,
        then quantizes to discrete codebook indices.

        Args:
            x: Input tensor of shape ``(B, C, *spatial)`` where:
                - B: batch size
                - C: number of channels (must match model's in_channels)
                - spatial: (H, W) for 2D or (H, W, D) for 3D

        Returns:
            Tuple of:
            - indices: Discrete codebook indices
            - quantized: Quantized continuous codes (for decoder)
            - loss: Quantization loss (commitment, entropy, etc.)

        Raises:
            TypeError: If x is not a floating point tensor
            ValueError: If x has wrong shape or contains NaN/Inf
        """
        validate_tensor_input(x, self.dim, self.config["in_channels"], "encode")

        h = self.encoder(x)
        h = self.quant_conv(h)

        # All quantizers now return (codes, loss, indices) consistently
        quantized, loss, indices = self.quantizer(h)

        return indices, quantized, loss

    def decode(
        self, quant: Float[torch.Tensor, "batch embedding_dim *spatial_compressed"]
    ) -> Float[torch.Tensor, "batch channels *spatial"]:
        """Decode from quantized continuous codes.

        Args:
            quant: Quantized codes from encode() (continuous representation).
                   Shape: ``(B, embedding_dim, *spatial_compressed)``

        Returns:
            Reconstructed output with original spatial dimensions

        Raises:
            TypeError: If quant is not a floating point tensor
            ValueError: If quant has wrong shape or contains NaN/Inf
        """
        validate_tensor_input(quant, self.dim, self.embedding_dim, "decode")

        quant = self.post_quant_conv(quant)
        return self.decoder(quant)

    def forward(
        self, input: Float[torch.Tensor, "batch channels *spatial"]
    ) -> dict[str, torch.Tensor] | NetworkEval:
        """Full forward pass: encode -> quantize -> decode.

        During training, returns dict with all outputs for loss computation.
        During evaluation, returns NetworkEval namedtuple.

        Args:
            input: Input tensor

        Returns:
            Training mode (dict):
                - 'reconstructions': Decoded output
                - 'quant_loss': Quantization loss
                - 'quant_info': Discrete indices
                - 'latents': Quantized codes (continuous)

            Eval mode (NetworkEval):
                - reconstructions: Decoded output
                - quant_loss: Quantization loss
                - quant_info: Discrete indices
        """
        indices, quant_codes, quant_loss = self.encode(input)
        reconstructions = self.decode(quant_codes)

        if self.training:
            return {
                "reconstructions": reconstructions,
                "quant_loss": quant_loss,
                "quant_info": indices,
                "latents": quant_codes,
            }

        return NetworkEval(
            reconstructions=reconstructions,
            quant_loss=quant_loss,
            quant_info=indices,
        )

    def tokenize(
        self, x: Float[torch.Tensor, "batch channels *spatial"]
    ) -> Int[torch.Tensor, "batch *spatial_indices"]:
        """Encode input to discrete token indices.

        This is the primary encoding method for inference and storage.
        Returns integer indices that can be stored efficiently or fed
        to autoregressive models.

        Args:
            x: Input tensor

        Returns:
            Discrete indices suitable for storage or sequence modeling
        """
        return self.encode(x)[0]

    def detokenize(
        self,
        indices: Int[torch.Tensor, "batch *spatial_indices"],
        spatial_shape: tuple[int, ...] | None = None,
    ) -> Float[torch.Tensor, "batch channels *spatial"]:
        """Decode from discrete token indices.

        Inverse of tokenize(). Converts discrete indices back to continuous
        output via codebook lookup and decoder. This is the canonical
        index-to-reconstruction decoding path for inference, converting
        stored/generated indices back to images.

        Args:
            indices: Discrete indices from tokenize(). Can be:
                - Spatial format: (B, H', W') for 2D or (B, H', W', D') for 3D
                - Flattened format: (B, N) where N = H' * W' [* D']
            spatial_shape: Original latent spatial dimensions (H', W') or (H', W', D').
                          Required when indices are flattened to avoid incorrect
                          cubic assumptions for anisotropic volumes.

        Returns:
            Reconstructed output

        Raises:
            ValueError: If spatial_shape is required but not provided
        """
        # RESFSQ indices are (B, Q, *spatial), the rest (B, *spatial). VQ and FSQ return channels-last
        # codes, LFQ and RESFSQ channels-first; the layout cannot be inferred from shape when a latent
        # side equals embedding_dim.
        spatial_axis = 2 if self.quantizer_type == "RESFSQ" else 1
        if indices.ndim == spatial_axis + 1 and self.dim > 1:
            if spatial_shape is None:
                raise ValueError(
                    "spatial_shape is required to detokenize flattened indices. "
                    "This prevents incorrect assumptions about spatial dimensions "
                    "for anisotropic volumes (e.g., medical images with non-cubic shapes)."
                )
            indices = indices.unflatten(spatial_axis, tuple(spatial_shape))

        quant = self.quantizer.indices_to_codes(indices)
        if self.quantizer_type in ("VQ", "FSQ"):
            quant = quant.movedim(-1, 1)
        quant = self.post_quant_conv(quant)
        return self.decoder(quant)

    def get_latent_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculate discrete index shape for given input shape.

        Note: Returns shape WITHOUT embedding dimension (just spatial).
        For RESFSQ, includes num_quantizers dimension.

        Args:
            input_shape: Input tensor shape (B, C, H, W) or (B, C, H, W, D)

        Returns:
            Expected index shape (B, H', W') or (B, H', W', D')
            where spatial dims are compressed by spatial_compression
        """
        b = input_shape[0]
        compression = self.spatial_compression

        if self.dim == 2:
            h, w = input_shape[2], input_shape[3]
            return (b, h // compression, w // compression)
        else:
            h, w, d = input_shape[2], input_shape[3], input_shape[4]
            return (b, h // compression, w // compression, d // compression)

    def get_codebook_size(self) -> int:
        """Get the total vocabulary size.

        Returns:
            Number of discrete codes in vocabulary
        """
        return self.quantizer.get_codebook_size()

    @torch.inference_mode()
    def reconstruct(
        self,
        x: torch.Tensor,
        roi_size: tuple[int, ...] | Optional[int] = None,
        overlap: float = 0.0,
    ) -> torch.Tensor:
        """Reconstruction with optional sliding window (overlap must be 0.0).

        Discrete tokenizers don't support overlapping windows because
        averaging discrete codes is not meaningful.

        Args:
            x: Input tensor
            roi_size: Window size for sliding window inference
            overlap: Must be 0.0 for discrete tokenizers

        Returns:
            Reconstructed output

        Raises:
            ValueError: If overlap > 0.0
        """
        if roi_size is not None and overlap > 0.0:
            raise ValueError(
                f"Discrete tokenizers require overlap=0.0 (got {overlap}). "
                "Use continuous tokenizer for overlapping reconstruction."
            )
        return super().reconstruct(x, roi_size=roi_size, overlap=overlap)
