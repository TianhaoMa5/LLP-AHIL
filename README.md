# README

This repository provides the implementation for a model training framework with various customizable parameters and configurations. Below are the descriptions of each parameter that can be set using command-line arguments:

### Dataset and Directory Configuration
- **`--root`**: Specifies the dataset directory (default: `./data`).
- **`--dataset`**: Name of the dataset used (default: `Cifar10`).
- **`--n-classes`**: Number of classes in the dataset (default: `10`).
- **`--folds`**: Specifies the number of dataset folds (default: `2`).

### Model Architecture
- **`--wresnet-k`**: Width factor for the Wide ResNet model (default: `2`).
- **`--wresnet-n`**: Depth of the Wide ResNet model (default: `28`).

### Training Configuration
- **`--n-epoches`**: Number of training epochs (default: `1024`).
- **`--batchsize`**: Batch size for training bag samples (default: `64`).
- **`--bagsize`**: Bag size of samples in each batch (default: `16`).
- **`--lr`**: Learning rate for training (default: `0.05`).
- **`--weight-decay`**: Weight decay for the optimizer (default: `1e-4`).
- **`--momentum`**: Momentum factor for the optimizer (default: `0.9`).
- **`--seed`**: Seed for random behaviors (default: `3`). Use negative values to disable seeding.

### Loss Coefficients
- **`--lam-u`**: Coefficient for unlabeled loss (default: `0.5`).
- **`--lam-c`**: Coefficient for contrastive loss (default: `1`).
- **`--lam-p`**: Coefficient for proportion loss (default: `2`).

### Exponential Moving Average (EMA)
- **`--eval-ema`**: Specifies whether to use the EMA model for evaluation (default: `True`).
- **`--ema-m`**: EMA momentum factor (default: `0.999`).

### Beta Parameters
- **`--beta-b`**: Beta parameter `b` (default: `5.0`).
- **`--beta-i`**: Beta parameter `i` (default: `5.0`).

### Experiment and Checkpoints
- **`--exp-dir`**: Experiment identifier (default: `L^2P-AHIL`).
- **`--checkpoint`**: Path to the pretrained model checkpoint (default: empty string).

---

### Usage Example

Run the script with default configurations:
```bash
python train.py
```

Customize parameters as needed:
```bash
python train.py --root './my_data' --dataset 'CIFAR-100' --n-classes 100 --batchsize 128 --lr 0.01
```

For more details, refer to the parameter descriptions above.