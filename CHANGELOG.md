# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-09-25

### Fixed

- `VectorQuantizer.get_codebook_entry` (and so `indices_to_codes` and
  `DiscreteTokenizer.detokenize`) now L2-normalises codebook entries when
  `use_norm=True`, matching `forward`. Previously, decoding VQ tokens from a
  `use_norm=True` model fed the decoder unnormalised embeddings.
- `DiscreteTokenizer.detokenize` now places the channel axis by quantizer type
  instead of inferring it from the tensor shape. Latents with a side equal to
  `embedding_dim` were decoded transposed, and flattened LFQ indices were
  reshaped incorrectly.

### Removed

- `DiscreteTokenizer._reshape_quant`; `detokenize` reshapes flattened indices
  (given `spatial_shape`) before the codebook lookup.

## [0.1.1] - 2026-09-25

### Changed

- The paper is linked from the README (arXiv badge), from `pyproject.toml`
  (`Paper` project URL), and from `CITATION.cff`, whose preferred citation is
  now the NeurIPS 2026 conference paper (arXiv:2608.07713). The README BibTeX
  entry is updated to match.
- The README points to
  [tokenizer-generator-coupling](https://github.com/liamchalcroft/tokenizer-generator-coupling),
  the repository holding the paper's experiments.

## [0.1.0] - 2026-08-07

First public release, accompanying the paper *Tokenizer-Generator Coupling in
Medical Image Generation*.

### Added

- Continuous tokenizers (`ContinuousTokenizer`, AE and VAE formulations) and
  discrete tokenizers (`DiscreteTokenizer`, VQ/FSQ/LFQ/ResidualFSQ heads) over a
  shared encoder-decoder backbone, for 2D images and 3D volumes.
- `MAISITokenizer` (fixed NVIDIA MAISI configuration), `TiTokTokenizer` (1D
  transformer tokenizer producing a fixed-length sequence), and the experimental
  `RAETokenizer` (frozen foundation-model encoder).
- Training infrastructure: trainer, reconstruction and adversarial losses, patch,
  multiscale and StyleGAN discriminators, callbacks, and NaN tracking.
- Evaluation: PSNR, SSIM, LPIPS, perplexity, and codebook usage, with
  `TokenizerEvaluator` to run them over a loader.
- Dataset tokenization to per-split `.npz` files via
  `scripts/tokenize_dataset.py`, plus `save_indices` / `load_indices` and
  `save_latents` / `load_latents`.
- HuggingFace Hub integration through `from_pretrained()` and upload helpers.
- A simulated BrainWeb T1-weighted brain volume bundled inside the package and
  resolved by `example_volume_path()`, so the volumetric example and tests run
  from an installed wheel without a download. The volume is simulated rather
  than acquired data; provenance and citation terms are in
  `src/medtokenizers/assets/README.md`.
- `py.typed` marker, so downstream type checkers consume the library's hints.
- MIT `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md`, with per-file SPDX
  attribution on code derived from CompVis latent-diffusion,
  lucidrains/vector-quantize-pytorch, NVIDIA Cosmos-Tokenizer, and MONAI/MAISI.
- Continuous integration across Python 3.10 to 3.12 with lint, test, and
  coverage lanes, a `.pre-commit-config.yaml`, and community health files
  (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`).

### Security

- All `torch.load` calls use `weights_only=True`. Checkpoints from untrusted
  sources should still be treated as untrusted input; see `SECURITY.md`.
