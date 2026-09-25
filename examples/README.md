# Examples

This folder contains example scripts for using medtokenizers.

## Available Examples

| Script | Description |
|--------|-------------|
| `train_medmnist2d.py` | Train a tokenizer on 2D MedMNIST datasets |
| `train_medmnist3d.py` | Train a tokenizer on 3D MedMNIST datasets |
| `train_with_callbacks.py` | Example using the Trainer API with callbacks |
| `evaluation_demo.py` | Evaluate a trained tokenizer |
| `inference_on_brain.py` | Run inference on brain MRI with sliding windows |

## Quick Start

```bash
# Install medmnist
pip install medmnist

# Train a 2D tokenizer
python examples/train_medmnist2d.py --dataset pathmnist --epochs 50

# Train a 3D tokenizer
python examples/train_medmnist3d.py --dataset organmnist3d --epochs 100

# Train a 3D continuous tokenizer (AE/VAE)
python examples/train_medmnist3d.py --dataset organmnist3d --tokenizer continuous --formulation VAE --epochs 100

# Run inference on a brain MRI
python examples/inference_on_brain.py --input brain.nii --model ./checkpoints/my-model
```

## MedMNIST Datasets

### 2D Datasets
- `pathmnist`, `chestmnist`, `dermamnist`, `octmnist`, `pneumoniamnist`
- `retinamnist`, `breastmnist`, `bloodmnist`, `tissuemnist`
- `organamnist`, `organcmnist`, `organsmnist`

### 3D Datasets
- `organmnist3d`, `nodulemnist3d`, `adrenalmnist3d`
- `fracturemnist3d`, `vesselmnist3d`, `synapsemnist3d`

## Quantization Methods

All examples support multiple quantization methods:

| Method | Description |
|--------|-------------|
| `VQ` | Vector Quantization (learned codebook) |
| `FSQ` | Finite Scalar Quantization |
| `LFQ` | Lookup-Free Quantization |
| `RESFSQ` | Residual FSQ (hierarchical) |

Example:
```bash
python examples/train_medmnist2d.py --dataset pathmnist --quantizer FSQ --levels 8 5 5 5
```

Both `train_medmnist2d.py` and `train_medmnist3d.py` also support continuous
tokenizers via `--tokenizer continuous --formulation {AE,VAE}`.
