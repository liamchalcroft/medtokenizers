# Copyright 2026 Liam Chalcroft
# SPDX-License-Identifier: MIT
#
# This file contains code derived from lucidrains/vector-quantize-pytorch
# (https://github.com/lucidrains/vector-quantize-pytorch), originally licensed under the MIT License.
# The FSQuantizer (bound/quantize/codes_to_indices/indices_to_codes with _levels and _basis
# buffers), ResidualFSQuantizer, and LFQuantizer (binary codebook + entropy regularization)
# follow that implementation, which itself adapts the JAX reference from Mentzer et al.,
# "Finite Scalar Quantization: VQ-VAE Made Simple" (arXiv:2309.15505).
# See THIRD_PARTY_NOTICES.md for details.
"""Quantization modules for discrete latent representations.

This module implements the quantization layer of VQ-VAE and related architectures.
Quantization is the key differentiator between continuous (VAE) and discrete
(VQ-VAE, FSQ) tokenizers - it maps continuous encoder outputs to a finite
vocabulary of codes.

The Quantization Problem
------------------------
Given encoder output z ∈ ℝ^d, find the nearest code in a codebook C = {c₁, ..., cₖ}:

    q(z) = argmin_{c ∈ C} ||z - c||₂

This is non-differentiable (argmin has zero gradient), so we use the
straight-through estimator (STE) to enable gradient flow during training.

Available Quantizers
--------------------
- VectorQuantizer: Classic learned codebook with commitment loss
- FSQuantizer: Implicit codebook via bounded scalar quantization
- ResidualFSQuantizer: Stacked FSQ for hierarchical refinement
- LFQuantizer: Binary codebook with entropy regularization

Index dtype
-----------
Output indices use int32 by default (supports vocab up to 2B).
For efficient storage, save with int16 (up to 32K codes) using
medtokenizers.inference.save_indices().

References:
    van den Oord et al. "Neural Discrete Representation Learning" (VQ-VAE)
    Mentzer et al. "Finite Scalar Quantization: VQ-VAE Made Simple" (FSQ)
    Yu et al. "Language Model Beats Diffusion" (LFQ in MagViT-2)
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from einops import pack, rearrange, reduce, unpack
from jaxtyping import Float, Int

from .base import BaseQuantizer
from .utils import entropy, jaxtyped_compile_safe, round_ste

logger = logging.getLogger(__name__)


class ResidualFSQuantizer(BaseQuantizer):
    """Residual Finite Scalar Quantization - Hierarchical Implicit Codebook.

    RESFSQ extends FSQ by stacking multiple quantizers in a residual manner,
    where each subsequent quantizer encodes the residual error from previous
    stages. This enables much larger effective codebook sizes while maintaining
    FSQ's simplicity.

    The Residual Principle
    ----------------------
    Given input z and N quantizers Q₁, Q₂, ..., Qₙ:

        r₀ = z                          # Initial residual
        z₁ = Q₁(r₀)                     # First quantization
        r₁ = r₀ - sg(z₁)                # Residual (stop-gradient)
        z₂ = Q₂(r₁)                     # Second quantization
        ...
        output = z₁ + z₂ + ... + zₙ     # Sum of all quantizations

    The stop-gradient (sg) on z_i when computing residuals ensures each
    quantizer learns to encode what previous ones missed, not to compete.

    Effective Codebook Size
    -----------------------
    If each FSQ has K codes, then N stacked quantizers have K^N effective codes.
    Example: 3 quantizers with [8,8,8] levels (512 codes each) = 512³ ≈ 134M codes.

    This exponential scaling is why RESFSQ can match or exceed VQ codebook
    expressiveness while retaining FSQ's implicit codebook benefits.

    Args:
        levels: Quantization levels per dimension for each FSQ layer.
                Example: [8, 8, 8] gives 512 codes per quantizer.
        num_quantizers: Number of residual quantization stages.
        embedding_dim: Input/output channel dimension. If different from
                      len(levels), linear projections are added.

    Example:
        >>> # 512^4 ≈ 68 billion effective codes
        >>> resfsq = ResidualFSQuantizer(
        ...     levels=[8, 8, 8],
        ...     num_quantizers=4,
        ...     embedding_dim=256
        ... )
        >>> z = torch.randn(2, 256, 16, 16, 16)
        >>> indices, codes, loss = resfsq(z)
        >>> indices.shape  # (2, 4, 16, 16, 16) - one index per quantizer
    """

    def __init__(
        self,
        levels: list[int],
        num_quantizers: int,
        embedding_dim: Optional[int] = None,
        share_projections: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dtype = kwargs.get("dtype", torch.float32)
        self.num_quantizers = num_quantizers
        self.levels = levels
        self.share_projections = share_projections

        if (
            share_projections
            and embedding_dim is not None
            and embedding_dim != len(levels)
        ):
            # Create shared projections once
            codebook_dim = len(levels)
            self.shared_project_in = nn.Linear(embedding_dim, codebook_dim)
            self.shared_project_out = nn.Linear(codebook_dim, embedding_dim)
            # Create FSQ layers without individual projections
            self.layers = nn.ModuleList(
                [
                    FSQuantizer(levels=levels, embedding_dim=None)
                    for _ in range(num_quantizers)
                ]
            )
        else:
            self.shared_project_in = None
            self.shared_project_out = None
            self.layers = nn.ModuleList(
                [
                    FSQuantizer(levels=levels, embedding_dim=embedding_dim)
                    for _ in range(num_quantizers)
                ]
            )

    @jaxtyped_compile_safe(beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch channels *spatial"]
    ) -> tuple[
        Float[torch.Tensor, "batch channels *spatial"],
        Float[torch.Tensor, "batch *spatial"],
        Int[torch.Tensor, "batch num_quantizers *spatial"],
    ]:
        """Apply residual quantization through all layers.

        AMP Precision Guards
        --------------------
        With many quantizers (e.g., 8+), repeated accumulation in float16 can:
        1. Accumulate rounding errors in the residual
        2. Overflow in quantized_out sum

        We compute residuals and accumulation in float32 to prevent these issues,
        then cast back to the input dtype at the end.

        Args:
            x: Input tensor from encoder, shape (B, C, ...) where ... is spatial dims

        Returns:
            quantized: Sum of all quantized outputs, shape (B, C, ...)
            loss: Accumulated loss (always zero for FSQ, kept for API consistency)
            indices: Stacked quantization indices, shape (B, num_quantizers, ...)

        Note:
            Return order is (codes, loss, indices) for consistency with
            VectorQuantizer, LFQuantizer, and FSQuantizer.
        """
        # Compute residuals and accumulation in float32 for precision
        # This prevents overflow/underflow with many quantizers under AMP
        residual = x.float()

        # Apply shared input projection if enabled
        if self.share_projections and self.shared_project_in is not None:
            # Project once: (B, C, *spatial) -> (B, codebook_dim, *spatial)
            residual = rearrange(residual, "b c ... -> b ... c")
            residual = self.shared_project_in(residual)
            residual = rearrange(residual, "b ... c -> b c ...")

        quantized_out: torch.Tensor = torch.zeros_like(residual)
        loss_out: torch.Tensor = torch.zeros(
            x.shape[0], *x.shape[2:], device=x.device, dtype=torch.float32
        )

        # Preallocate indices tensor to avoid Python list stack
        # Use int32 for efficient storage (supports vocab up to 2B)
        indices = torch.empty(
            (x.shape[0], self.num_quantizers, *x.shape[2:]),
            device=x.device,
            dtype=torch.int32,
        )

        for idx, layer in enumerate(self.layers):
            # FSQuantizer returns (codes, loss, indices) after standardization
            z, loss, quant_indices = layer(residual)

            # Stop gradient on z when computing residual
            residual = residual - z.detach()
            quantized_out = quantized_out + z
            loss_out = loss_out + loss.squeeze(dim=1)
            indices[:, idx] = quant_indices

        # Apply shared output projection if enabled
        if self.share_projections and self.shared_project_out is not None:
            quantized_out = rearrange(quantized_out, "b c ... -> b ... c")
            quantized_out = self.shared_project_out(quantized_out)
            quantized_out = rearrange(quantized_out, "b ... c -> b c ...")

        # Return order: (codes, loss, indices) for consistency with VQ/LFQ/FSQ
        return quantized_out.to(self.dtype), loss_out.to(self.dtype), indices

    @jaxtyped_compile_safe(beartype)
    def indices_to_codes(
        self, indices_stack: Int[torch.Tensor, "batch num_quantizers *spatial"]
    ) -> Float[torch.Tensor, "batch channels *spatial"]:
        """Decode stacked indices back to continuous codes.

        Args:
            indices_stack: Quantization indices from each layer

        Returns:
            Sum of decoded codes from all layers, shape ``(B, C, *spatial)``
        """
        quantized_out: torch.Tensor | int = 0
        for layer, indices in zip(self.layers, indices_stack.transpose(0, 1)):
            # FSQuantizer.indices_to_codes returns (*batch, channels)
            # We need to accumulate in that format, then permute at the end
            quantized_out = quantized_out + layer.indices_to_codes(indices)
        # Move channels from last dimension to position 1: (B, *spatial, C) -> (B, C, *spatial)
        quantized_out = quantized_out.movedim(-1, 1)

        # Apply shared output projection if enabled
        if self.share_projections and self.shared_project_out is not None:
            quantized_out = rearrange(quantized_out, "b c ... -> b ... c")
            quantized_out = self.shared_project_out(quantized_out)
            quantized_out = rearrange(quantized_out, "b ... c -> b c ...")

        return quantized_out

    def get_codebook_size(self) -> int:
        """Get effective codebook size (product of all layer sizes)."""
        return self.layers[0].codebook_size ** len(self.layers)


class FSQuantizer(BaseQuantizer):
    """Finite Scalar Quantization - Implicit Codebook via Bounded Rounding.

    FSQ represents a paradigm shift from traditional vector quantization:
    instead of learning an explicit codebook of embedding vectors, FSQ
    implicitly defines the codebook as the Cartesian product of scalar
    quantization levels.

    The Mathematical Foundation
    ---------------------------
    For a d-dimensional latent space with levels L = [L₁, L₂, ..., L_d],
    the implicit codebook has exactly ∏Lᵢ entries. Each entry is defined
    by its integer coordinates within the level bounds:

        code(i₁, i₂, ..., i_d) = [2i₁/(L₁-1) - 1, ..., 2i_d/(L_d-1) - 1]

    This means the codebook is known a priori, never needs gradient updates,
    and can be arbitrarily large without increasing parameter count.

    Why FSQ Over VQ?
    ----------------
    1. **No codebook collapse**: Every code is equally accessible by design
    2. **Simpler training**: No auxiliary losses (commitment, entropy)
    3. **Deterministic encoding**: Same input always maps to same code
    4. **Memory efficiency**: Implicit codebook uses O(d) parameters, not O(K×d)

    The Trade-off
    -------------
    FSQ trades codebook expressiveness for stability. Each dimension is
    quantized independently, losing the ability to learn correlated
    quantization boundaries. For most tokenization tasks, this is acceptable.

    AMP Stability (Critical)
    ------------------------
    The bound() function uses tanh for soft clamping. Under float16 AMP:
    - atanh() is undefined at ``|x| >= 1``, causing NaN
    - tanh() saturates at extremes, causing vanishing gradients

    This implementation uses float32 intermediate computation and careful
    clamping to prevent these failure modes.

    Args:
        levels: List of integers specifying quantization levels per dimension.
                Example: [8, 8, 8, 8] gives 4096 implicit codes.
        embedding_dim: Input/output channel dimension. If different from
                      len(levels), linear projections are automatically added.
        num_codebooks: For multi-codebook FSQ (rarely needed). Default: 1.
        scale: Optional output scaling factor.

    Example:
        >>> # 4096-code FSQ for 256-dim latents
        >>> fsq = FSQuantizer(levels=[8, 8, 8, 8], embedding_dim=256)
        >>> z = torch.randn(2, 256, 16, 16, 16)  # (B, C, H, W, D)
        >>> indices, codes, loss = fsq(z)
        >>> # indices: (B, H, W, D) - integer codes
        >>> # codes: (B, C, H, W, D) - quantized latents
        >>> # loss: dummy zero tensor (FSQ has no auxiliary loss)

    References:
        Mentzer et al. "Finite Scalar Quantization: VQ-VAE Made Simple"
        https://arxiv.org/abs/2309.15505
    """

    def __init__(
        self,
        levels: list[int],
        embedding_dim: Optional[int] = None,
        num_codebooks: int = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dtype = kwargs.get("dtype", torch.float32)

        # Register levels and basis as buffers for device/dtype handling
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.int64))
        self.register_buffer(
            "_basis",
            torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.int64), dim=0),
        )
        # Pre-compute half_width as float buffer for quantization
        self.register_buffer(
            "_half_width",
            (torch.tensor(levels, dtype=torch.int64) // 2).float(),
        )
        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim
        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = (
            keep_num_codebooks_dim
            if keep_num_codebooks_dim is not None
            else num_codebooks > 1
        )
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.embedding_dim = (
            embedding_dim if embedding_dim is not None else len(levels) * num_codebooks
        )

        # Add projection layers if dimensions don't match
        has_projections = self.embedding_dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.embedding_dim, effective_codebook_dim)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.embedding_dim)
            if has_projections
            else nn.Identity()
        )

        # Compute codebook size (can be large, use int64)
        self.codebook_size = int(self._levels.prod().item())

        # Lazy codebook generation - don't precompute for large codebooks
        self._cached_implicit_codebook: Optional[torch.Tensor] = None

    @property
    def implicit_codebook(self) -> torch.Tensor:
        """Lazily compute and cache implicit codebook on first access.

        For large codebook sizes (>100K), this avoids allocating memory
        until actually needed.
        """
        if self._cached_implicit_codebook is None:
            with torch.no_grad():
                indices = torch.arange(
                    self.codebook_size, device=self._levels.device, dtype=torch.int64
                )
                self._cached_implicit_codebook = self.indices_to_codes(
                    indices, project_out=False
                )
        return self._cached_implicit_codebook

    def get_codebook_size(self) -> int:
        """Return total number of codes in implicit codebook."""
        return self.codebook_size

    def bound(
        self, z: Float[torch.Tensor, "*batch codebook_dim"], eps: float = 1e-3
    ) -> Float[torch.Tensor, "*batch codebook_dim"]:
        """Softly bound input values to quantization level range.

        Uses tanh for smooth, differentiable bounding that allows gradients
        to flow while ensuring outputs stay within valid quantization range.

        AMP Safety
        ----------
        This function is carefully implemented to avoid numerical issues
        under float16 automatic mixed precision:

        1. Explicit autocast disable to guarantee float32 computation
        2. atanh input clamped to (-1, 1) to prevent domain errors
        3. tanh input clamped to prevent saturation and vanishing gradients

        The autocast disable is CRITICAL: even though we call .float(), PyTorch's
        autocast can still affect intermediate operations. The explicit disable
        ensures all ops (atanh, tanh, etc.) run in float32.

        Args:
            z: Unbounded input values
            eps: Small margin to prevent exact boundary values

        Returns:
            Bounded values in range suitable for quantization
        """
        # CRITICAL: Explicitly disable autocast to guarantee float32 computation.
        # This prevents subtle AMP bugs where intermediate ops still use float16.
        device_type = "cuda" if z.is_cuda else "cpu"
        with torch.amp.autocast(device_type, enabled=False):
            z_f32 = z.float()
            levels_f32 = self._levels.float()

            half_l = (levels_f32 - 1) * (1 + eps) / 2
            offset = torch.where(self._levels % 2 == 0, 0.5, 0.0).float()

            # Clamp ratio to prevent atanh domain errors (|x| must be < 1)
            ratio = (offset / half_l).clamp(-0.9999, 0.9999)
            shift = ratio.atanh()

            # Replace any NaN unconditionally (avoids data-dependent branch for torch.compile)
            # The nan_to_num is a no-op if there are no NaNs
            shift = torch.nan_to_num(shift, nan=0.0)

            # Log warning outside of compile for debugging (data-dependent, so guard it)
            if not torch.compiler.is_compiling():
                if torch.isnan(ratio.atanh()).any():
                    logger.warning(
                        "FSQuantizer.bound(): NaN detected in atanh output despite clamping. "
                        "This may indicate numerical instability. Replacing NaN with 0.0. "
                        "Input stats: min=%.4f, max=%.4f, has_nan=%s",
                        z_f32.min().item(),
                        z_f32.max().item(),
                        torch.isnan(z_f32).any().item(),
                    )

            # Clamp tanh input to prevent saturation (tanh(±6) ≈ ±1)
            bounded_input = (z_f32 + shift).clamp(-6.0, 6.0)
            result = bounded_input.tanh() * half_l - offset

        return result.to(z.dtype)

    def quantize(
        self, z: Float[torch.Tensor, "*batch codebook_dim"]
    ) -> Float[torch.Tensor, "*batch codebook_dim"]:
        """Quantize bounded values to discrete levels.

        Applies rounding with straight-through estimator (STE) for
        gradient flow, then normalizes to [-1, 1] range.

        Args:
            z: Input values (should be bounded first)

        Returns:
            Quantized values normalized to approximately [-1, 1]
        """
        quantized = round_ste(self.bound(z))
        return quantized / self._half_width

    def _scale_and_shift(
        self, zhat_normalized: Float[torch.Tensor, "*batch codebook_dim"]
    ) -> Float[torch.Tensor, "*batch codebook_dim"]:
        """Convert normalized codes to integer indices for codebook lookup."""
        return (zhat_normalized * self._half_width) + self._half_width

    def _scale_and_shift_inverse(
        self, zhat: Float[torch.Tensor, "*batch codebook_dim"]
    ) -> Float[torch.Tensor, "*batch codebook_dim"]:
        """Convert integer indices back to normalized codes."""
        return (zhat - self._half_width) / self._half_width

    def codes_to_indices(
        self, zhat: Float[torch.Tensor, "*batch codebook_dim"]
    ) -> Int[torch.Tensor, "*batch"]:
        """Convert quantized codes to flat codebook indices.

        Maps multi-dimensional code coordinates to a single integer index
        using mixed-radix encoding: idx = Σ (z_i * basis_i)

        Args:
            zhat: Quantized code coordinates

        Returns:
            Single integer index per code
        """
        zhat_shifted = self._scale_and_shift(zhat)
        # Use int64 arithmetic to prevent precision loss for large codebooks
        zhat_int = zhat_shifted.round().to(torch.int64)
        return (zhat_int * self._basis).sum(dim=-1).to(torch.int32)

    @overload
    def indices_to_codes(
        self, indices: Int[torch.Tensor, "*batch"], project_out: Literal[True] = True
    ) -> Float[torch.Tensor, "*batch embedding_dim"]: ...

    @overload
    def indices_to_codes(
        self, indices: Int[torch.Tensor, "*batch"], project_out: Literal[False]
    ) -> Float[torch.Tensor, "*batch codebook_dim"]: ...

    def indices_to_codes(
        self, indices: Int[torch.Tensor, "*batch"], project_out: bool = True
    ) -> Float[torch.Tensor, "*batch channels"]:
        """Convert flat indices back to continuous codes.

        Inverse of codes_to_indices: extracts multi-dimensional code
        coordinates from flat index using modular arithmetic.

        Args:
            indices: Flat codebook indices
            project_out: Whether to apply output projection (for embedding_dim != codebook_dim)

        Returns:
            Continuous code values, optionally projected to embedding_dim
        """
        indices = rearrange(indices, "... -> ... 1")
        # Use int64 throughout to prevent overflow for large codebooks
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered.float())

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        if project_out:
            codes = self.project_out(codes)

        return codes.to(self.dtype)

    @jaxtyped_compile_safe(beartype)
    def forward(
        self, z: Float[torch.Tensor, "batch channels *spatial"]
    ) -> tuple[
        Float[torch.Tensor, "batch channels *spatial"],
        Float[torch.Tensor, "batch 1 *spatial"],
        Int[torch.Tensor, "batch *spatial"],
    ]:
        """Quantize encoder output to discrete codes.

        Args:
            z: Encoder output, shape (B, C, H, W) for 2D or (B, C, H, W, D) for 3D

        Returns:
            out: Quantized output, same shape as input
            loss: Dummy zero loss (FSQ has no auxiliary loss)
            indices: Codebook indices, shape (B, H, W) or (B, H, W, D)

        Note:
            Return order is (codes, loss, indices) for consistency with
            VectorQuantizer and LFQuantizer.
        """
        # Detect dimensionality from input shape
        ndim = len(z.shape)
        if ndim == 4:
            dim = 2
            b, c, h, w = z.shape
            assert c == self.embedding_dim, (
                f"Input channels {c} must match embedding_dim {self.embedding_dim}"
            )
            z_flat = rearrange(z, "b c h w -> b (h w) c")
            spatial_shape = (h, w)
        elif ndim == 5:
            dim = 3
            b, c, h, w, d = z.shape
            assert c == self.embedding_dim, (
                f"Input channels {c} must match embedding_dim {self.embedding_dim}"
            )
            z_flat = rearrange(z, "b c h w d -> b (h w d) c")
            spatial_shape = (h, w, d)
        else:
            raise ValueError(f"Expected 4D or 5D input, got {ndim}D")

        # Project to codebook dimension if needed
        z_proj = self.project_in(z_flat)
        z_proj = rearrange(z_proj, "b n (c d) -> b n c d", c=self.num_codebooks)

        # Quantize
        codes = self.quantize(z_proj)
        indices = self.codes_to_indices(codes)

        # Project back to embedding dimension
        codes = rearrange(codes, "b n c d -> b n (c d)")
        out = self.project_out(codes)

        # Reshape to spatial dimensions
        if dim == 2:
            out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        else:
            out = rearrange(out, "b (h w d) c -> b c h w d", h=h, w=w, d=d)

        # Remove codebook dimension from indices if single codebook
        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        # Reshape indices to spatial dimensions
        if dim == 2:
            indices = rearrange(indices, "b (h w) -> b h w", h=h, w=w)
        else:
            indices = rearrange(indices, "b (h w d) -> b h w d", h=h, w=w, d=d)

        # Dummy loss (FSQ has no auxiliary loss, but API expects one)
        loss_shape = [b, 1] + list(spatial_shape)
        dummy_loss = torch.zeros(loss_shape, device=z.device, dtype=z.dtype)

        # Return order: (codes, loss, indices) for consistency with VQ/LFQ
        return out.to(self.dtype), dummy_loss, indices


class VectorQuantizer(BaseQuantizer):
    """Vector Quantization with Learnable Codebook.

    The classic VQ-VAE quantization layer that learns an explicit codebook
    of K embedding vectors. Each encoder output is mapped to its nearest
    codebook entry, enabling discrete latent representations.

    The VQ Objective
    ----------------
    Given encoder output z and codebook {e₁, ..., eₖ}, find nearest code:

        q(z) = eₖ where k = argmin_i ||z - eᵢ||₂

    Since argmin is non-differentiable, we use the straight-through estimator:
    - Forward: output = nearest codebook entry
    - Backward: gradient flows directly to encoder (as if output = z)

    Loss Components
    ---------------
    1. **Codebook loss**: ||sg[z] - e||₂² - moves codes toward encoder outputs
    2. **Commitment loss**: ||z - sg[e]||₂² - moves encoder toward codes

    The β hyperparameter balances these: L = L_codebook + β * L_commitment

    Codebook Collapse Prevention
    ----------------------------
    A common failure mode where only a few codes are ever selected. Signs:
    - Low codebook utilization (perplexity << K)
    - Some embeddings never update

    This implementation provides two mitigations:

    1. **EMA codebook updates** (use_ema=True): Instead of gradient-based updates,
       the codebook is updated via exponential moving average of encoder outputs.
       This is more stable and often leads to better codebook utilization.

    2. **Dead code reset** (reset_unused_codes=True): Codes that haven't been
       used in `dead_code_threshold` batches are re-initialized to random
       encoder outputs, keeping the full codebook active.

    Args:
        num_embeddings: Codebook size (K). Larger = more expressive but harder to train.
        embedding_dim: Dimension of each embedding vector.
        dim: Spatial dimension (1, 2, or 3). Auto-detected if None.
        beta: Commitment loss weight. Default 0.25.
        sane_index_shape: If True, return indices without trailing dimension.
        use_norm: If True, normalize embeddings (cosine similarity matching).
        use_ema: If True, use EMA codebook updates instead of gradients. Default False.
        ema_decay: EMA decay rate. Higher = slower updates. Default 0.99.
        reset_unused_codes: If True, reset codes that haven't been used. Default False.
        dead_code_threshold: Number of batches before a code is considered dead. Default 100.

    Example:
        >>> vq = VectorQuantizer(num_embeddings=1024, embedding_dim=256, use_ema=True)
        >>> z = torch.randn(2, 256, 16, 16, 16)  # Encoder output
        >>> z_q, loss, indices = vq(z)
        >>> # z_q: quantized output, same shape as z
        >>> # loss: codebook + commitment loss
        >>> # indices: (2, 16, 16, 16) - which code was selected

    References:
        van den Oord et al. "Neural Discrete Representation Learning"
        https://arxiv.org/abs/1711.00937
    """

    # Chunk size for distance computation. Vectors are processed in chunks of this
    # size to reduce peak memory usage. 32K balances memory efficiency with throughput.
    _DISTANCE_CHUNK_SIZE: int = 32768

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dim: int,
        beta: float = 0.25,
        sane_index_shape: bool = True,
        use_norm: bool = False,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        reset_unused_codes: bool = False,
        dead_code_threshold: int = 100,
        **kwargs,
    ) -> None:
        super().__init__()
        if dim not in (1, 2, 3):
            raise ValueError(
                f"VectorQuantizer.dim must be 1, 2, or 3, got {dim}. "
                "Pass dim=1 for (B,C,L), dim=2 for (B,C,H,W), or dim=3 for (B,C,H,W,D)."
            )
        self.n_e = num_embeddings
        self.e_dim = embedding_dim
        self.beta = beta
        self.use_norm = use_norm
        self.dtype = kwargs.get("dtype", torch.float32)
        self.dim = dim
        self.sane_index_shape = sane_index_shape

        # EMA codebook update settings
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.reset_unused_codes = reset_unused_codes
        self.dead_code_threshold = dead_code_threshold

        # Initialize codebook with small uniform values
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        if use_ema:
            # EMA codebook: don't update embedding weights via gradient
            self.embedding.weight.requires_grad = False

            # Track cluster sizes and embedding sums for EMA updates
            self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
            self.register_buffer("ema_embed_sum", self.embedding.weight.data.clone())

            # Track batches since each code was last used (for dead code detection)
            self.register_buffer("batches_since_used", torch.zeros(num_embeddings))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Optionally L2-normalize for cosine similarity matching."""
        if self.use_norm:
            return F.normalize(x, dim=-1)
        return x

    def _find_nearest_embedding(self, z_flattened: torch.Tensor) -> torch.Tensor:
        """Find nearest codebook entry for each vector.

        Uses chunked computation to reduce peak memory when the number of
        vectors is large. The full distance matrix for N vectors and K
        embeddings requires O(N×K) memory, which becomes problematic for
        3D volumes (e.g., 128K vectors × 1K embeddings = 0.5GB).

        Chunked processing reduces peak memory to O(chunk_size×K) while
        maintaining identical output. On CPU, this also improves cache
        utilization for a modest speedup.

        Args:
            z_flattened: Input vectors, shape (N, embedding_dim)

        Returns:
            Indices of nearest embeddings, shape (N,)
        """
        if self.use_norm:
            z_flattened = F.normalize(z_flattened, dim=-1)
            embeddings = F.normalize(self.embedding.weight, dim=-1)
        else:
            embeddings = self.embedding.weight

        n_vectors = z_flattened.shape[0]
        chunk_size = self._DISTANCE_CHUNK_SIZE

        # For small inputs, compute full distance matrix (simpler, same speed)
        if n_vectors <= chunk_size:
            z_sq = z_flattened.pow(2).sum(dim=1, keepdim=True)
            e_sq = embeddings.pow(2).sum(dim=1, keepdim=True).t()
            d = z_sq + e_sq - 2 * z_flattened @ embeddings.t()
            return torch.argmin(d, dim=1)

        # Chunked computation for large inputs
        encoding_indices = torch.empty(
            n_vectors, device=z_flattened.device, dtype=torch.long
        )
        # Pre-compute embedding squared norms (constant across chunks)
        e_sq = embeddings.pow(2).sum(dim=1)  # (K,)

        for start in range(0, n_vectors, chunk_size):
            end = min(start + chunk_size, n_vectors)
            chunk = z_flattened[start:end]

            # ||a - b||² = ||a||² + ||b||² - 2*<a,b>
            z_sq = chunk.pow(2).sum(dim=1)  # (chunk_size,)
            d = z_sq.unsqueeze(1) + e_sq.unsqueeze(0) - 2 * chunk @ embeddings.t()
            encoding_indices[start:end] = torch.argmin(d, dim=1)

        return encoding_indices

    def get_codebook_size(self) -> int:
        """Return number of codebook entries."""
        return self.n_e

    def _update_ema(
        self, z_flattened: torch.Tensor, encoding_indices: torch.Tensor
    ) -> None:
        """Update codebook via exponential moving average.

        EMA Update Rule
        ---------------
        For each code k:
            N_k ← γ * N_k + (1-γ) * n_k           # Update count
            m_k ← γ * m_k + (1-γ) * Σ z_i[k=k]    # Update sum
            e_k ← m_k / N_k                        # Normalize

        Where:
            γ = ema_decay
            n_k = number of vectors assigned to code k in this batch
            z_i[k=k] = vectors assigned to code k

        This is more stable than gradient updates because:
        1. All codes get some update signal (via decay)
        2. Updates are proportional to usage
        3. No optimizer state to manage

        Distributed Training
        --------------------
        When using DistributedDataParallel (DDP), each worker sees different
        batches. The EMA statistics must be synchronized across workers to
        keep codebooks consistent. We all-reduce cluster_size and embed_sum
        before the EMA update.

        Note: This adds communication overhead but is essential for correctness.
        Without synchronization, codebooks diverge across workers after ~100 steps.
        """
        if not self.training:
            return

        with torch.no_grad():
            # One-hot encoding of assignments
            encodings = F.one_hot(encoding_indices, self.n_e).float()

            # Count assignments per code
            cluster_size = encodings.sum(dim=0)

            # Sum of vectors assigned to each code
            embed_sum = encodings.t() @ z_flattened

            # Distributed training: synchronize statistics across workers
            # This ensures all workers have consistent codebooks
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(cluster_size)
                torch.distributed.all_reduce(embed_sum)

            # EMA update (in-place for efficiency)
            self.ema_cluster_size.mul_(self.ema_decay).add_(
                cluster_size, alpha=1 - self.ema_decay
            )
            self.ema_embed_sum.mul_(self.ema_decay).add_(
                embed_sum, alpha=1 - self.ema_decay
            )

            # Update embeddings: e_k = m_k / N_k
            # Use clamp instead of addition to prevent division by small numbers in float16
            # 1e-3 is safe for float16 (min positive ~6e-8) and prevents precision issues
            n = self.ema_cluster_size.unsqueeze(1).clamp(min=1e-3)
            self.embedding.weight.data.copy_(self.ema_embed_sum / n)

            # Track code usage for dead code detection (in-place)
            used_codes = cluster_size > 0
            self.batches_since_used[used_codes] = 0
            self.batches_since_used[~used_codes] += 1

            # Reset dead codes
            if self.reset_unused_codes:
                self._reset_dead_codes(z_flattened)

    def _reset_dead_codes(self, z_flattened: torch.Tensor) -> None:
        """Reset codes that haven't been used for many batches.

        Dead codes are re-initialized to random encoder outputs from the
        current batch, giving them a chance to be selected again.

        Note: This method uses .item() and torch.randperm() which are not
        compatible with torch.compile(fullgraph=True) or torch.vmap.
        It should only be called outside compiled regions (guarded at callsite).
        """
        # Defense-in-depth: skip if called during compilation (should not happen
        # since _update_ema callsite is guarded, but protects against future misuse)
        if torch.compiler.is_compiling():
            return

        dead_codes = self.batches_since_used > self.dead_code_threshold

        if not dead_codes.any():
            return

        num_dead = dead_codes.sum().item()

        # Sample random encoder outputs to replace dead codes
        # Note: randperm is non-deterministic; keep out of compiled/vmap regions
        random_indices = torch.randperm(z_flattened.size(0))[:num_dead]
        new_embeddings = (
            z_flattened[random_indices].detach().to(self.embedding.weight.dtype)
        )

        # Replace dead codes
        dead_indices = dead_codes.nonzero(as_tuple=True)[0]
        self.embedding.weight.data[dead_indices] = new_embeddings

        # Reset EMA stats for these codes
        self.ema_cluster_size.data[dead_indices] = 1.0
        self.ema_embed_sum.data[dead_indices] = new_embeddings
        self.batches_since_used.data[dead_indices] = 0

    def get_codebook_usage(self) -> dict[str, float]:
        """Get codebook utilization statistics.

        Returns:
            Dictionary with:
            - 'perplexity': Exponential of entropy (higher = more uniform usage)
            - 'usage_fraction': Fraction of codes used at least once
            - 'dead_codes': Number of codes never used
        """
        if not self.use_ema:
            return {"perplexity": 0.0, "usage_fraction": 0.0, "dead_codes": 0.0}

        # Normalize cluster sizes to get usage distribution
        total = self.ema_cluster_size.sum()
        if total < 1e-5:
            return {
                "perplexity": 0.0,
                "usage_fraction": 0.0,
                "dead_codes": float(self.n_e),
            }

        probs = self.ema_cluster_size / total
        entropy = -(probs * (probs + 1e-10).log()).sum()
        perplexity = entropy.exp().item()

        usage_fraction = (self.ema_cluster_size > 1e-5).float().mean().item()
        dead_codes = (self.batches_since_used > self.dead_code_threshold).sum().item()

        return {
            "perplexity": perplexity,
            "usage_fraction": usage_fraction,
            "dead_codes": dead_codes,
        }

    @jaxtyped_compile_safe(beartype)
    def forward(
        self,
        z: Float[torch.Tensor, "batch channels *spatial"],
        temp: Optional[float] = None,
        rescale_logits: bool = False,
        return_logits: bool = False,
    ) -> tuple[
        Float[torch.Tensor, "batch channels *spatial"],
        Float[torch.Tensor, "batch *spatial_loss"],
        Int[torch.Tensor, "batch *spatial"],
    ]:
        """Quantize input to nearest codebook entry.

        Args:
            z: Encoder output, shape (B, C, L), (B, C, H, W), or (B, C, H, W, D)
            temp: Temperature for soft quantization (unused, for API compatibility)
            rescale_logits: Whether to rescale distance logits (unused)
            return_logits: Whether to return distance logits (unused)

        Returns:
            z_q: Quantized output with straight-through gradient
            loss: Combined codebook + commitment loss
            indices: Selected codebook indices
        """
        # dim is validated at init time (must be 2 or 3)
        dim = self.dim

        # Rearrange to (batch, spatial..., channels)
        if dim == 1:
            z = rearrange(z, "b c l -> b l c")
        elif dim == 2:
            z = rearrange(z, "b c h w -> b h w c")
        else:
            z = rearrange(z, "b c h w d -> b h w d c")

        batch_spatial_shape = z.shape[:-1]
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # Find nearest codebook entry (uses chunked computation for large inputs)
        encoding_indices = self._find_nearest_embedding(z_flattened)

        # Lookup quantized vectors
        z_q = self.embedding(encoding_indices).view(z.shape)

        # Apply optional normalization
        z_q = self._normalize(z_q)
        z = self._normalize(z)

        # EMA update skipped during torch.compile for fullgraph compatibility
        if self.use_ema and self.training and not torch.compiler.is_compiling():
            self._update_ema(z_flattened, encoding_indices)

        # Compute losses (per-sample, keeping spatial dims)
        if dim == 1:
            reduce_dims = [1, 2]  # l, c
        elif dim == 2:
            reduce_dims = [1, 2, 3]  # h, w, c
        else:
            reduce_dims = [1, 2, 3, 4]  # h, w, d, c

        if self.use_ema:
            # With EMA, we only need commitment loss (encoder → codebook)
            # The codebook is updated via EMA, not gradients
            loss = self.beta * torch.mean(
                (z_q.detach() - z).square(), dim=reduce_dims, keepdim=True
            )
        else:
            # Standard VQ loss: codebook + commitment
            # Codebook loss: ||sg[z] - e||² (moves codes toward encoder outputs)
            codebook_loss = torch.mean(
                (z_q - z.detach()).square(), dim=reduce_dims, keepdim=True
            )

            # Commitment loss: ||z - sg[e]||² (moves encoder toward codes)
            commitment_loss = torch.mean(
                (z_q.detach() - z).square(), dim=reduce_dims, keepdim=True
            )

            loss = codebook_loss + self.beta * commitment_loss

        # Straight-through estimator: gradient flows to z, but forward uses z_q
        z_q = z + (z_q - z).detach()

        # Reshape back to (batch, channels, spatial...)
        if dim == 1:
            z_q = rearrange(z_q, "b l c -> b c l")
        elif dim == 2:
            z_q = rearrange(z_q, "b h w c -> b c h w")
        else:
            z_q = rearrange(z_q, "b h w d c -> b c h w d")

        # Reshape indices to spatial shape
        if self.sane_index_shape:
            encoding_indices = encoding_indices.view(batch_spatial_shape)

        # Use int32 for efficient storage (supports vocab up to 2B)
        return z_q.to(self.dtype), loss, encoding_indices.to(torch.int32)

    def indices_to_codes(
        self,
        indices: Int[torch.Tensor, "*batch"],
        shape: tuple[int, ...] | None = None,
    ) -> Float[torch.Tensor, "*batch embedding_dim"]:
        """Convert indices back to codebook embeddings.

        Args:
            indices: Codebook indices
            shape: Optional shape to reshape output

        Returns:
            Codebook embeddings for given indices
        """
        return self.get_codebook_entry(indices, shape)

    def get_codebook_entry(
        self,
        indices: Int[torch.Tensor, "*batch"],
        shape: tuple[int, ...] | None = None,
    ) -> Float[torch.Tensor, ...]:
        """Lookup codebook entries by index.

        Args:
            indices: Codebook indices (flat or spatial)
            shape: Target shape for output

        Returns:
            Codebook embeddings, reshaped if shape provided
        """
        z_q = self._normalize(self.embedding(indices))

        if shape is not None:
            z_q = z_q.view(shape)
            if self.dim == 1:
                z_q = z_q.permute(0, 2, 1).contiguous()
            elif self.dim == 2:
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            elif self.dim == 3:
                z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        return z_q.to(self.dtype)


class LFQuantizer(BaseQuantizer):
    """Lookup-Free Quantization - Binary Codebook with Entropy Regularization.

    LFQ takes the implicit codebook idea to its extreme: the codebook is simply
    all 2^d binary vectors in {-1, +1}^d. No learning, no storage - just sign
    quantization with optional entropy regularization.

    The LFQ Philosophy
    ------------------
    Why learn a codebook when binary codes are universal? Any continuous vector
    can be approximated by its sign pattern, and with enough dimensions, this
    provides sufficient expressiveness.

    The codebook is implicitly defined as:
        C = {c ∈ {-1, +1}^d : all binary sign vectors}

    Quantization is simply: q(z) = sign(z)

    Entropy Regularization
    ----------------------
    To prevent codebook collapse (always selecting the same codes), LFQ can add
    entropy regularization that encourages uniform code usage:

        L_entropy = H(avg_prob) - avg(H(prob))

    where prob is the soft assignment probability over codes.

    This pushes the model toward using all available codes equally.

    Args:
        codebook_size: Target codebook size (must be power of 2)
        codebook_dim: Dimension of binary codes (= log2(codebook_size))
        dim: Spatial dimension (2 or 3)
        num_codebooks: Number of independent codebooks (rarely used)
        embedding_dim: Input/output dimension (projects to/from codebook_dim)
        entropy_loss_weight: Weight for entropy regularization
        commitment_loss_weight: Weight for commitment loss
        default_temp: Temperature for soft entropy computation
        entropy_loss: Whether to apply entropy regularization

    Example:
        >>> # 256-code LFQ (8-bit binary codes)
        >>> lfq = LFQuantizer(
        ...     codebook_size=256,
        ...     codebook_dim=8,
        ...     embedding_dim=64,
        ...     entropy_loss=True
        ... )
        >>> z = torch.randn(2, 64, 16, 16, 16)
        >>> z_q, loss, indices = lfq(z)

    References:
        Yu et al. "Language Model Beats Diffusion: Tokenizer is Key to Visual Generation"
        https://arxiv.org/abs/2310.05737
    """

    def __init__(
        self,
        *,
        codebook_size: int,
        codebook_dim: int,
        dim: int,
        num_codebooks: int = 1,
        embedding_dim: Optional[int] = None,
        entropy_loss_weight: float = 0.1,
        commitment_loss_weight: float = 0.25,
        default_temp: float = 0.01,
        entropy_loss: bool = False,
        entropy_sample_size: int = 8192,
        **kwargs,
    ) -> None:
        super().__init__()
        if dim not in (2, 3):
            raise ValueError(
                f"LFQuantizer.dim must be 2 or 3, got {dim}. "
                "Pass dim=2 for 2D (B,C,H,W) or dim=3 for 3D (B,C,H,W,D) inputs."
            )
        if entropy_loss and num_codebooks > 1:
            raise ValueError(
                "LFQuantizer entropy regularization requires num_codebooks == 1, got "
                f"num_codebooks={num_codebooks}. The entropy codebook is single-codebook "
                "by construction (its mask spans the full codebook_dim and its enumerated "
                "codebook has codebook_size entries of codebook_dim bits), so it cannot "
                "match the per-codebook width used when num_codebooks > 1. Either set "
                "num_codebooks=1 or disable entropy_loss."
            )
        self.entropy_loss = entropy_loss
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks
        self.default_temp = default_temp
        self.entropy_loss_weight = entropy_loss_weight
        self.commitment_loss_weight = commitment_loss_weight
        self.entropy_sample_size = entropy_sample_size
        self.dtype = kwargs.get("dtype", torch.float32)
        self.dim = dim

        embedding_dim = embedding_dim or codebook_dim
        has_projections = embedding_dim != codebook_dim
        self.project_in = (
            nn.Linear(embedding_dim, codebook_dim) if has_projections else nn.Identity()
        )
        self.project_out = (
            nn.Linear(codebook_dim, embedding_dim) if has_projections else nn.Identity()
        )

        if entropy_loss:
            self.codebook_size = codebook_size

            # Use int64 to prevent overflow for codebook_dim > 30
            self.register_buffer(
                "mask", 2 ** torch.arange(codebook_dim - 1, -1, -1, dtype=torch.int64)
            )

            all_codes = torch.arange(codebook_size, dtype=torch.int64)
            bits = ((all_codes[..., None] & self.mask) != 0).float()
            self.register_buffer("codebook", 2 * bits - 1.0)

    @jaxtyped_compile_safe(beartype)
    def forward(
        self,
        z: Float[torch.Tensor, "batch channels *spatial"],
        temp: Optional[float] = None,
    ) -> tuple[
        Float[torch.Tensor, "batch channels *spatial"],
        Float[torch.Tensor, "batch ..."],
        Int[torch.Tensor, "batch ..."],
    ]:
        """Quantize input using binary sign function.

        Args:
            z: Encoder output
            temp: Temperature for soft entropy (uses default_temp if None)

        Returns:
            z_q: Binary quantized output with STE gradient
            loss: Commitment + optional entropy loss
            indices: Binary code indices
        """
        temp = temp or self.default_temp

        # dim is validated at init time (must be 2 or 3)
        dim = self.dim

        # Rearrange to (batch, spatial, channels)
        z = rearrange(z, "b d ... -> b ... d")
        z, ps = pack([z], "b * d")
        z = self.project_in(z)
        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        # Binary quantization: q(z) = sign(z) with STE
        original_input = z
        z_q = torch.where(z > 0, 1.0, -1.0)  # Scalars broadcast, no temporary tensor
        z_q = z + (z_q - z).detach()  # STE

        # Commitment loss: encourage encoder to output near ±1
        commit_loss = (original_input - z_q.detach()).square().mean(dim=[1, 2, 3])

        # Reshape output
        z_q = rearrange(z_q, "b n c d -> b n (c d)")
        z_q = self.project_out(z_q)
        z_q = unpack(z_q, ps, "b * d")[0]
        z_q = rearrange(z_q, "b ... d -> b d ...")

        loss = self.commitment_loss_weight * commit_loss

        # Entropy regularization
        if self.entropy_loss:
            # Compute indices from binary pattern
            indices = reduce(
                (original_input > 0).long() * self.mask, "b n c d -> b n c", "sum"
            )
            indices = unpack(indices, ps, "b * c")[0]
            indices = rearrange(indices, "... 1 -> ...")

            # Soft assignment probabilities for entropy
            # Sample for large batches to avoid O(N×2^d) memory/compute
            n_vectors = original_input.shape[0] * original_input.shape[1]
            if n_vectors > self.entropy_sample_size:
                flat_input = rearrange(original_input, "b n c d -> (b n) c d")
                sample_idx = torch.randperm(
                    flat_input.shape[0], device=flat_input.device
                )[: self.entropy_sample_size]
                sampled_input = flat_input[sample_idx]
                distance = -2 * torch.einsum(
                    "n c d, j d -> n c j",
                    sampled_input,
                    self.codebook.to(sampled_input.dtype),
                )
                prob = (-distance / temp).softmax(dim=-1)
                per_sample_entropy = entropy(prob).mean()
                avg_prob = prob.mean(dim=0)
                codebook_entropy = entropy(avg_prob).mean()
            else:
                distance = -2 * torch.einsum(
                    "... i d, j d -> ... i j",
                    original_input,
                    self.codebook.to(original_input.dtype),
                )
                prob = (-distance / temp).softmax(dim=-1)
                per_sample_entropy = entropy(prob).mean(dim=[1, 2])
                avg_prob = reduce(prob, "... c d -> c d", "mean")
                codebook_entropy = entropy(avg_prob).mean()

            entropy_aux_loss = per_sample_entropy - codebook_entropy

            loss = loss + self.entropy_loss_weight * entropy_aux_loss

            # Reshape loss to match expected output shape
            loss_shape = loss.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            if dim == 3:
                loss_shape = loss_shape.unsqueeze(1)

            # Use int32 for efficient storage (supports vocab up to 2B)
            return z_q.to(self.dtype), loss_shape, indices.to(torch.int32)

        # Generate indices without entropy loss
        d_size = original_input.shape[-1]
        mask = 2 ** torch.arange(
            d_size - 1, -1, -1, device=original_input.device, dtype=torch.int64
        )
        indices = reduce((original_input > 0).long() * mask, "b n c d -> b n c", "sum")

        # Unpack indices to spatial shape (consistent with entropy_loss=True path)
        indices = unpack(indices, ps, "b * c")[0]
        # Remove codebook dim if single codebook (consistent with other quantizers)
        if self.num_codebooks == 1:
            indices = rearrange(indices, "... 1 -> ...")

        # Reshape loss
        loss_shape = loss.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        if dim == 3:
            loss_shape = loss_shape.unsqueeze(1)

        # Use int32 for efficient storage (supports vocab up to 2B)
        return z_q.to(self.dtype), loss_shape, indices.to(torch.int32)

    def get_codebook_size(self) -> int:
        """Return codebook size (2^codebook_dim)."""
        if hasattr(self, "codebook_size"):
            return self.codebook_size
        return 2**self.codebook_dim

    def indices_to_codes(
        self, indices: Int[torch.Tensor, "..."]
    ) -> Float[torch.Tensor, "..."]:
        """Convert indices to binary codes.

        For single codebook: indices shape ``[B, *spatial]``, returns ``[B, C, *spatial]``.
        For multi-codebook: indices shape ``[B, *spatial, num_codebooks]``,
        returns ``[B, C, *spatial]`` (codebook dim is consumed).

        Args:
            indices: Integer indices representing binary patterns

        Returns:
            Continuous codes in channels-first format
        """
        d_size = self.codebook_dim // self.num_codebooks

        if self.num_codebooks > 1:
            # Multi-codebook: indices [B, *spatial, num_codebooks], each index encodes d_size bits
            mask = 2 ** torch.arange(
                d_size - 1, -1, -1, device=indices.device, dtype=torch.int64
            )
            codes = ((indices.unsqueeze(-1) & mask) != 0).float()
            codes = 2 * codes - 1.0
            # [B, *spatial, num_codebooks, d_size] -> [B, *spatial, codebook_dim]
            codes = codes.flatten(-2)
        elif hasattr(self, "codebook"):
            codes = self.codebook[indices]
        else:
            # Single codebook: each index encodes codebook_dim bits
            mask = 2 ** torch.arange(
                self.codebook_dim - 1, -1, -1, device=indices.device, dtype=torch.int64
            )
            codes = ((indices.unsqueeze(-1) & mask) != 0).float()
            codes = 2 * codes - 1.0

        # project_out returns (batch, *spatial, channels), need (batch, channels, *spatial)
        codes = self.project_out(codes).to(self.dtype)
        return codes.movedim(-1, 1)
