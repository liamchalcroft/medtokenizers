Choosing a Tokenizer
====================

medtokenizers provides several tokenizer families, all subclasses of
:class:`~medtokenizers.BaseTokenizer` with a common interface.

Continuous vs. discrete
-----------------------

- :class:`~medtokenizers.ContinuousTokenizer` -- VAE/AE-style models that map a
  scan to a continuous latent grid. Pair these with continuous generative models
  (diffusion, flow matching).
- :class:`~medtokenizers.DiscreteTokenizer` -- VQ / FSQ / LFQ / ResidualFSQ models
  that map a scan to a grid of integer codes. Pair these with discrete generators
  (autoregressive, MaskGIT, discrete diffusion) in
  `medlatents <https://github.com/liamchalcroft/medlatents>`_.

Specialized tokenizers
----------------------

- :class:`~medtokenizers.MAISITokenizer` -- a continuous tokenizer configured to
  the NVIDIA MAISI VAE layout, with sliding-window ``reconstruct`` for large 3D
  volumes.
- :class:`~medtokenizers.TiTokTokenizer` -- a transformer tokenizer that encodes
  an image or volume into a fixed-length 1D sequence of ``K`` tokens.
- :class:`~medtokenizers.RAETokenizer` *(experimental)* -- a Representation
  Autoencoder pairing a frozen foundation-model encoder with a trainable patch
  decoder.

Rule of thumb
-------------

Choose a discrete tokenizer if you intend to train discrete generators; choose a
continuous tokenizer for continuous diffusion / flow models. TiTok is useful when
a compact 1D sequence is preferable to a spatial token grid.
