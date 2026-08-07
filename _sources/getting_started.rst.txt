Getting Started
===============

Installation
------------

.. code-block:: bash

   pip install -e .
   pip install -e ".[dev]"    # tests + linting
   pip install -e ".[docs]"   # documentation
   pip install -e ".[all]"    # everything (cloud + training too)

medtokenizers requires Python 3.10+ and PyTorch 2.0+.

Reconstruct a volume (MAISI)
----------------------------

A simulated T1-weighted brain volume from the BrainWeb Simulated Brain Database
ships inside the package, so the examples run from an installed wheel as well as
a source checkout. ``example_volume_path()`` resolves it; see
``medtokenizers/assets/README.md`` for provenance. Random initialization works
for a smoke test; load your own trained weights for real reconstructions.

.. code-block:: python

   import medrs
   import torch
   from medtokenizers import MAISITokenizer, example_volume_path

   img = medrs.load(str(example_volume_path()))
   volume = img.to_torch_with_dtype_and_device(dtype=torch.float32)
   volume = volume.unsqueeze(0).unsqueeze(0)
   volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

   model = MAISITokenizer().eval()  # or MAISITokenizer.from_pretrained("./checkpoint")
   with torch.no_grad():
       recon = model.reconstruct(volume, roi_size=(96, 96, 64), overlap=0.5)

Discrete tokens (TiTok)
-----------------------

.. code-block:: python

   import torch
   from medtokenizers import TiTokTokenizer

   tok = TiTokTokenizer(
       dim=2, in_channels=1, out_channels=1, resolution=64,
       patch_size=16, num_tokens=32, num_embeddings=1024, hidden_dim=256,
   )
   x = torch.randn(2, 1, 64, 64)
   tokens = tok.tokenize(x)        # (B, K) integer codes
   recon = tok.detokenize(tokens)  # back to image space

These token grids are exactly what `medlatents
<https://github.com/liamchalcroft/medlatents>`_ consumes to train generative
models -- medlatents calls ``tokenizer.detokenize(...)`` to turn generated tokens
back into images.

Evaluation
----------

.. code-block:: python

   from medtokenizers import load_tokenizer, TokenizerEvaluator

   model = load_tokenizer("./path/to/checkpoint")
   evaluator = TokenizerEvaluator(model, device="cuda")
   results = evaluator.evaluate(test_loader)
   evaluator.print_results(results)
