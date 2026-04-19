
Submission for 
Project — 10-class robust image classification on 32×32 RGB images.

**Goal**: Maximize **Macro F1** across 10 classes under distribution shift (train ≠ test).
**Goal**: Maximize **Macro F1** across 10 classes under extreme class imbalance and distribution shift (train ≠ test).

## Overview

The repository features an "All-in-One Training + Inference" script (`notebook.py`, v3). It replaces the previous multi-phase pipeline with a simplified, highly effective single-script approach that trains from scratch and generates `submission.csv`.

## Architecture & Methods in Detail

To combat extreme class imbalances and testing-time distribution shifts without external data or pre-trained internet models, the challenge pipeline employs several customized methods:

### 1. Model Architecture
- **WideResNet-28-10 (WRN)**: A high-capacity Residual Network variant specifically constructed for 32x32 image inputs natively. 
- With a depth of 28 layers and a widening factor of 10, it features approximately **~36 Million parameters**. Its wider blocks and Dropout layers (p=0.3) provide significant representational capacity whilst regularizing against overfitting.

### 2. Imbalance Handling
- **Sqrt-Inverse Frequency Sampling**: A Weighted Random Sampler handles training loop stochasticity by heavily oversampling the minority (tail) classes. Unlike standard inverse frequency, weights are calculated as `1.0 / sqrt(class_counts)`, which balances uniformity without entirely neglecting the head classes.
- **Balanced Softmax Loss**: The network tackles the long-tailed class distribution through a specialized Balanced Softmax Loss. Logits are actively shifted by `log(class_prior)` during the forward pass, neutralizing the classifier's inherent bias towards majority class predictions. 

### 3. Advanced Data Augmentations
- **Train Transforms**: `AutoAugment` (CIFAR-10 Policy) combined with `Cutout` (16x16 pixel masking), Random Cropping, and Horizontal Flipping to ensure extensive spatial and color invariances.
- **MixUp & CutMix Regularization**: Employed stochastically (50% probability) within the training loop with `alpha=1.0` to interpolate images and labels directly. This forces the model to learn smoother decision boundaries and linearly robust features between classes.

### 4. Optimization & Convergence
- **Optimization Strategy**: Trained with SGD (Stochastic Gradient Descent) featuring Nesterov Momentum (0.9) and high weight decay (`5e-4`). 
- **Cosine Annealing with Warmup**: A tailored Lambda LR scheduler initiates with 5 warmup epochs to gracefully stabilize initial gradients, organically decaying over the total epochs using a cosine curve.
- **Stochastic Weight Averaging (SWA)**: After standard convergence (Epoch 360), SWA begins caching weight states alongside a flat `SWALR`. Final batch-norm statistics are updated natively, flattening out the local minima curves to drastically improve underlying test generalization.

### 5. Multi-Ensemble Inference
- **3-Seed Ensembling**: Models are fully trained from scratch across three distinct pseudo-random seeds (42, 137, 7). The Softmax probability distributions are then numerically averaged during inference.
- **Aggressive Test-Time Augmentation (TTA)**: During inference, for every single image, the network generates predictions for **1 clean view + 30 augmented views** per seed model. TTA incorporates dynamic Random Rotations (10 degrees), Crop, Flip, and Color Jittering to completely stabilize shifting prediction confidence scores.

---

## Setup

### 1. Install Dependencies

```bash
pip install torch torchvision pyyaml scikit-learn pandas numpy tqdm pillow
pip install torch torchvision numpy scikit-learn pillow
```

> For GPU training, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/):
> For GPU training, install PyTorch with CUDA (example for CUDA 11.8):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 2. Dataset

Place the competition data so the structure looks like:

```
Shiftguard10/
├── shift-guard-10-robust-image-classification-challenge/
│   ├── classes.txt
│   ├── train_labels.csv
│   ├── sample_submission.csv
│   ├── train_images/   (29,400 PNGs)
│   └── test_images/    (7,600 PNGs)
├── src/
├── configs/
└── README.md
```
Ensure the competition data is available on your machine (e.g., from Kaggle). You can specify the path with `--data-root` when running the notebook. By default, it looks for common paths or the `shift-guard-10-robust-image-classification-challenge/` directory in the current working directory.

## Training
## Usage

We use a robust 3-Phase **Supervised Contrastive Learning (SupCon)** approach to maximize F1 under extreme class imbalance, trained completely from scratch.
The entire pipeline is driven by `notebook.py`. 

### Supported Architectures
You can use `--model` or `--backbone` with: `cct`, `wrn`, `resnet50`, `convnext`, `effnet`.

### Quick Debug (CPU, ~2 min)
```bash
python src/supcon.py --backbone resnet50 --debug
python src/train.py --model resnet50 --debug
```

### Full Training (GPU)

**Phase 1: SupCon Pretraining**
Learn highly invariant, clustered class features without the linear classifier bias.
### Quick Debug / Smoke Test
Run a fast smoke test (2 epochs, 1 seed, 2 TTA views) to ensure everything works:
```bash
python src/supcon.py --backbone resnet50 --balanced-sampling --epochs 300 --gpu 0
python notebook.py --debug
```
*(This saves the backbone state to `checkpoints/supcon_resnet50_epoch300.pth`)*

**Phase 2: Linear Probing**
Freeze the backbone and warm up only the classification head safely.
### Full Training & Inference
Run the complete pipeline (3 seeds × 450 epochs + 30 TTA views). This will generate `submission.csv` at the end:
```bash
python src/train.py --config configs/default.yaml --model resnet50 \
  --pretrained-backbone checkpoints/supcon_resnet50_epoch300.pth \
  --linear-probe --epochs 20 --gpu 0
python notebook.py --data-root /path/to/data
```

**Phase 3: End-to-End Fine-Tuning**
Unfreeze the entire network and fine-tune with a small learning rate.
### Advanced Training Configuration
You can customize hyperparameters to adjust training duration or ensemble size:
```bash
python src/train.py --config configs/default.yaml --model resnet50 \
  --pretrained-backbone checkpoints/supcon_resnet50_epoch300.pth \
  --epochs 100 --lr 0.01 --gpu 0
```
# Single seed training (faster, equivalent to v2)
python notebook.py --seeds 42

### Multi-GPU Execution
To train multiple architectures simultaneously on different GPUs:
```bash
# Terminal 1: Train ConvNeXt on GPU 0
python src/supcon.py --backbone convnext --balanced-sampling --gpu 0
# Custom epochs and TTA views
python notebook.py --epochs 300 --tta 20

# Terminal 2: Train EfficientNet on GPU 1
python src/supcon.py --backbone effnet --balanced-sampling --gpu 1
# Run on a specific GPU
python notebook.py --gpu 1
```

Checkpoints are saved to `checkpoints/`.

## Inference

### Inference Only
If you already have trained checkpoints (saved by default in the specified `--checkpoint-dir`), you can skip training and just generate predictions:
```bash
# Single model inference (runs on GPU 0 by default)
python src/inference.py --checkpoint checkpoints/best_cct.pth

# With Test-Time Augmentation (5 views) on specific GPU
python src/inference.py --checkpoint checkpoints/best_cct.pth --tta 5 --gpu 1

# Ensemble CCT + WRN with TTA
python src/inference.py --checkpoint checkpoints/best_cct.pth checkpoints/best_wrn.pth --tta 5
python notebook.py --inference-only
```

Output: `submission.csv` (7,600 rows with `id,label`).

## Project Structure

| File | Description |
|------|-------------|
| `src/dataset.py` | Dataset, augmentation (SupCon transforms, TrivialAugment, TTA) |
| `src/models/cct.py` | Compact Convolutional Transformer (CCT-7/3×1) |
| `src/models/wideresnet.py` | WideResNet-28-10 for CIFAR-style inputs |
| `src/models/torchvision_models.py` | ResNet50, ConvNeXt, EffNet natively adapted to 32x32 |
| `src/supcon.py` | Supervised Contrastive Pretraining phase |
| `src/train.py` | Fine-tuning loop — supports `--linear-probe` |
| `src/inference.py` | Inference with TTA and multi-model ensemble |
| `src/utils.py` | Metrics, MixUp/CutMix, checkpointing |
| `configs/default.yaml` | All hyperparameters |

## Key Techniques

To strictly abide by the competition rules (no pretrained models, no external data), our pipeline utilizes:
- **Supervised Contrastive Learning (SupCon)** — pulls all images of the same class together in representation space to robustly cluster minority classes *before* the linear classifier is initialized.
- **Frozen Linear Probing** — protects the pretrained features from destruction by the randomly initialized dense head, acting as a phase-break.
- **Architectural Diversity Ensembling** — leverages Transformers, Wide ResNets, standard ResNets, and ConvNeXts (custom adapted for 32x32 image structures) to ensure powerful generalization against the hidden distribution shift.
- **Enhanced TTA + SWA** — combines multi-checkpoint outputs with Random Rotation, Flip, Color Jitter, and Weight Averaging for ultimate test-time stability.
## Checkpoints
Checkpoints are saved automatically after each epoch and support resuming if training is interrupted. When using the ensemble setting with multiple seeds, models are saved separately as `wrn_seed<SEED>.pth` within the checkpoint directory.

## Output
Predictions are saved directly to `submission.csv` containing `id` and `label` columns, ready for competition submission.
