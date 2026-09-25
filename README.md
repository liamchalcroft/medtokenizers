# medtokenizers

[![arXiv](https://img.shields.io/badge/arXiv-2608.07713-b31b1b.svg)](https://arxiv.org/abs/2608.07713)

`medtokenizers` compresses 2D and 3D medical images into latents or discrete
token grids. It provides one encoder-decoder backbone with interchangeable
quantization heads (VQ, LFQ, FSQ, Residual FSQ, VAE, AE), so the quantizer can
be treated as a controlled variable rather than a fixed preprocessing choice.
It also provides shared reconstruction metrics and dataset tokenization
utilities.

This library accompanies the NeurIPS 2026 paper
[*Tokenizer-Generator Coupling in Medical Image Generation*](https://arxiv.org/abs/2608.07713),
which uses it for the tokenizer half of a factorial study. The paper's
experiments live in
[tokenizer-generator-coupling](https://github.com/liamchalcroft/tokenizer-generator-coupling).
See [Citation](#citation).

## Install

```bash
pip install medtokenizers
```

Requires Python 3.10+ and PyTorch 2.0+. Tested on Linux and macOS. A CUDA GPU
is needed for 3D training, but everything below runs on CPU.

For a CPU-only install, take PyTorch from its CPU index first so `torch` and
`torchvision` come from the same build:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install medtokenizers
```

To work on the library itself, install from a checkout instead:

```bash
git clone https://github.com/liamchalcroft/medtokenizers.git
cd medtokenizers
pip install -e ".[test]"
```

## Run it

Swap the quantizer, keep everything else fixed:

```python
import torch

from medtokenizers import DiscreteTokenizer

x = torch.randn(1, 1, 64, 64)

for quantizer in ["VQ", "FSQ", "RESFSQ"]:
    model = DiscreteTokenizer(
        dim=2, quantizer=quantizer, resolution=64, spatial_compression=8
    ).eval()
    with torch.no_grad():
        tokens = model.tokenize(x)  # (1, 8, 8) integer codes
        recon = model.detokenize(tokens)  # (1, 1, 64, 64)
```

LFQ is the one head that needs two further arguments, because its codebook is
defined by its binary dimension rather than inferred:

```python
model = DiscreteTokenizer(
    dim=2,
    quantizer="LFQ",
    resolution=64,
    spatial_compression=8,
    codebook_size=1024,
    codebook_dim=10,
)
```

The continuous heads live on a separate class, selected by `formulation`
instead of `quantizer`, and return a real-valued latent rather than indices:

```python
from medtokenizers import ContinuousTokenizer

model = ContinuousTokenizer(
    dim=2, formulation="VAE", resolution=64, spatial_compression=8
).eval()
with torch.no_grad():
    latent = model.tokenize(x)  # (1, 4, 8, 8)
    recon = model.detokenize(latent)
```

At evaluation time `ContinuousTokenizer.encode` returns the posterior mean, so
latents extracted with `model.eval()` are deterministic.

Set `dim=3` for volumes; the same calls take `(B, C, H, W, D)` tensors.

## Volumetric inference

A simulated T1-weighted brain volume ships inside the package (from the BrainWeb
Simulated Brain Database, with provenance in
[src/medtokenizers/assets/README.md](src/medtokenizers/assets/README.md)), so the
example runs without a download and works from an installed wheel.
`example_volume_path()` resolves it. Random initialisation is enough for a smoke
test; load your own weights for real reconstructions.

```bash
python examples/inference_on_brain.py --model-type maisi
```

Large volumes are reconstructed with Gaussian-weighted sliding windows:

```python
import medrs
import torch

from medtokenizers import MAISITokenizer, example_volume_path

img = medrs.load(str(example_volume_path()))
volume = img.to_torch_with_dtype_and_device(dtype=torch.float32)
volume = volume.unsqueeze(0).unsqueeze(0)  # (H, W, D) -> (1, 1, H, W, D)
volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

model = MAISITokenizer().eval()
with torch.no_grad():
    recon = model.reconstruct(volume, roi_size=(96, 96, 64), overlap=0.5)
```

Sliding-window reconstruction over a full 1 mm volume is memory-hungry. Reduce
`roi_size` or `overlap` if you are not on a GPU.

## Tokenizers

| Class | Selected by | Output |
| --- | --- | --- |
| `DiscreteTokenizer` | `quantizer="VQ" \| "FSQ" \| "LFQ" \| "RESFSQ"` | integer token grid |
| `ContinuousTokenizer` | `formulation="VAE" \| "AE"` | real-valued latent |
| `MAISITokenizer` | fixed MAISI configuration | real-valued latent |
| `TiTokTokenizer` | 1D transformer tokenizer | fixed-length sequence of K tokens |
| `RAETokenizer` *(experimental)* | frozen foundation-model encoder | real-valued latent |

Adding a quantizer means subclassing `BaseQuantizer` (two abstract methods,
`forward` and `indices_to_codes`) and wiring it into `DiscreteTokenizer`. The
encoder-decoder and the tokenization I/O are unchanged by the choice.

### TiTok

TiTok encodes an image or volume into a fixed-length 1D sequence of K tokens.
The encoder concatenates patch tokens with K learnable latent tokens and keeps
only the latter; the decoder reconstructs from quantized latents plus mask
tokens. Input shape must equal `resolution`, and each spatial dimension must be
divisible by `patch_size`.

```python
from medtokenizers import TiTokTokenizer

tokenizer = TiTokTokenizer(
    dim=2,
    in_channels=1,
    out_channels=1,
    resolution=64,
    patch_size=16,
    num_tokens=32,
    num_embeddings=1024,
    hidden_dim=256,
)
indices, codes, quant_loss = tokenizer.encode(torch.randn(2, 1, 64, 64))
recon = tokenizer.decode(codes)
```

TiTok's original two-stage proxy-code warmup and decoder fine-tuning are not
wired into the training scripts; implement that externally if you want the
published recipe.

## Evaluation

```python
from medtokenizers import compute_lpips, compute_psnr, compute_ssim

psnr = compute_psnr(reference, reconstruction)
ssim = compute_ssim(reference, reconstruction)
lpips = compute_lpips(reference, reconstruction)
```

`TokenizerEvaluator` runs these over a loader, together with perplexity and
codebook utilisation for discrete models:

```python
from medtokenizers import TokenizerEvaluator, load_tokenizer

model = load_tokenizer("./path/to/checkpoint")
results = TokenizerEvaluator(model, device="cuda").evaluate(test_loader)
```

No pretrained checkpoints ship with this repository.

## Ecosystem

`medtokenizers` is one of three packages:

- [medrs](https://github.com/liamchalcroft/med-rs): medical-image I/O
  (NIfTI/DICOM) and preprocessing primitives.
- [medtokenizers](https://github.com/liamchalcroft/medtokenizers): tokenizers
  that compress scans into latents or token grids *(this package)*.
- [medlatents](https://github.com/liamchalcroft/medlatents): generative models
  (autoregressive, MaskGIT, diffusion, flow matching, Bayesian flow) that learn
  over those tokens.

```text
images -> medrs (load) -> medtokenizers (tokenize) -> medlatents (generate)
       -> medtokenizers (detokenize) -> images
```

`scripts/tokenize_dataset.py` writes a directory of per-split `.npz` files plus
`metadata.json`, with key `codes` (int16) for discrete models and `latents`
(float16) for continuous ones.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and the
pull-request workflow, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for
community standards.

## Citation

```bibtex
@inproceedings{chalcroft2026tokenizer,
  title     = {Tokenizer--Generator Coupling in Medical Image Generation},
  author    = {Chalcroft, Liam},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {arXiv:2608.07713}
}

@software{chalcroft_medtokenizers,
  author  = {Chalcroft, Liam},
  title   = {{medtokenizers}: Continuous and discrete tokenizers for volumetric medical imaging},
  url     = {https://github.com/liamchalcroft/medtokenizers},
  version = {0.1.2}
}
```

Machine-readable metadata is in [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).

Some modules derive from third-party projects: CompVis (Stable Diffusion,
taming-transformers) and lucidrains `vector-quantize-pytorch` under MIT, and
NVIDIA Cosmos-Tokenizer and MONAI/MAISI under Apache-2.0. Their terms are
preserved in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[NOTICE](NOTICE). Two things are outside the MIT licence: NVIDIA MAISI
pretrained *weights*, if you use them, remain under NSCLv1; and the bundled
BrainWeb volume is data carrying its own citation requirement.
