medtokenizers
=============

Tokenization pipelines for volumetric medical imaging -- continuous and
discrete tokenizers (VAE/AE, VQ/FSQ/LFQ/ResidualFSQ, TiTok, RAE, MAISI) with
supporting quantization modules, training, and evaluation infrastructure.

Ecosystem
---------

``medtokenizers`` is one of three packages that together form a toolkit for
generative modeling of medical images:

- `medrs <https://github.com/liamchalcroft/med-rs>`_ -- fast medical-image I/O (NIfTI/DICOM) and preprocessing primitives.
- ``medtokenizers`` -- continuous and discrete tokenizers that compress 2D/3D scans into latents or token grids *(this package)*.
- `medlatents <https://github.com/liamchalcroft/medlatents>`_ -- generative models (autoregressive, MaskGIT, diffusion, flow matching, Bayesian flow) that learn over those tokens.

Typical pipeline::

   images -> medrs (load) -> medtokenizers (tokenize) -> medlatents (generate) -> medtokenizers (detokenize) -> images

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started

.. toctree::
   :maxdepth: 2
   :caption: Guides

   guides/tokenizers
   guides/training
   guides/evaluation

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/tokenizers
   api/quantizers
   api/evaluation
   api/training
   api/io

.. toctree::
   :maxdepth: 1
   :caption: Project Info

   contributing

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
