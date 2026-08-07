Tokenizers
==========

All tokenizers subclass :class:`~medtokenizers.BaseTokenizer` and share a common
interface (``encode`` / ``decode`` / ``tokenize`` / ``detokenize`` /
``reconstruct`` / ``forward`` / ``from_pretrained`` / ``save_pretrained``).

.. autoclass:: medtokenizers.BaseTokenizer
   :members:
   :show-inheritance:

.. autoclass:: medtokenizers.ContinuousTokenizer
   :members:
   :show-inheritance:

.. autoclass:: medtokenizers.DiscreteTokenizer
   :members:
   :show-inheritance:

.. autoclass:: medtokenizers.MAISITokenizer
   :members:
   :show-inheritance:

.. autoclass:: medtokenizers.TiTokTokenizer
   :members:
   :show-inheritance:

.. autoclass:: medtokenizers.RAETokenizer
   :members:
   :show-inheritance:
