Evaluating Tokenizers
=====================

:class:`~medtokenizers.TokenizerEvaluator` computes reconstruction and
codebook-quality metrics over a dataloader.

.. code-block:: python

   from medtokenizers import load_tokenizer, TokenizerEvaluator

   model = load_tokenizer("./path/to/checkpoint")
   evaluator = TokenizerEvaluator(model, device="cuda")
   results = evaluator.evaluate(test_loader)
   evaluator.print_results(results)

Available metrics include PSNR, SSIM, LPIPS, MSE / MAE, perplexity, and codebook
usage -- see :doc:`../api/evaluation`.
