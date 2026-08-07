Training Tokenizers
===================

medtokenizers ships a :class:`~medtokenizers.training.Trainer` with a callback
system for training tokenizers from scratch.

.. code-block:: bash

   python scripts/train.py --help

The trainer supports continuous (reconstruction + KL) and discrete
(reconstruction + quantization/commitment + optional entropy) objectives, mixed
precision, and checkpointing. Callbacks such as
:class:`~medtokenizers.training.EarlyStopping`,
:class:`~medtokenizers.training.Checkpoint`,
:class:`~medtokenizers.training.LRScheduler`, and
:class:`~medtokenizers.training.Logger` hook into the training loop.

See :doc:`../api/training` for the full API.
