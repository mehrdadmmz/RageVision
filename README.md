# RageVision

### CNN-Driven Emotion Detection in Streaming Content

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Model](https://img.shields.io/badge/Model-MobileNetV2-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-86%25-brightgreen)
![scikit-learn](https://img.shields.io/badge/Metrics-Scikit--learn-F7931E?logo=scikitlearn)
![NumPy](https://img.shields.io/badge/Lib-NumPy-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Viz-Matplotlib-11557c)

A binary image classifier that detects emotional rage outbursts in Twitch streaming content using **MobileNetV2** with transfer learning. Trained on ~10,400 labeled video frames, the model achieves **86% accuracy** with balanced precision and recall across both classes.

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [Experiments](#experiments)
- [Contributors](#contributors)

## Architecture

RageVision uses **MobileNetV2** as a feature extractor with a custom classification head. The base model's parameters are frozen (transfer learning), and only the classifier is trained on the rage detection task.

<img width="810" alt="MobileNetV2 Architecture" src="https://github.com/user-attachments/assets/d42b3e7f-158b-46b4-87a6-1a6036e7237f" />
<img width="719" alt="Transfer Learning Pipeline" src="https://github.com/user-attachments/assets/717d84a1-e33e-4846-a679-3cac7cb870cc" />

**Classification Head:**
```
MobileNetV2 (frozen) -> Dropout(0.3) -> Linear(1280 -> 2)
```

**Training Setup:**
| Component | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Loss | CrossEntropyLoss |
| Epochs | 10 |
| Batch Size | 32 |

## Project Structure

```
ragevision/
├── __init__.py
├── config.py                  # Hyperparameters and settings
├── data/
│   ├── dataset.py             # Data loading and augmentation pipeline
│   └── frame_extractor.py     # Video-to-frame extraction utility
├── models/
│   └── classifier.py          # MobileNetV2 model builder
├── training/
│   ├── trainer.py             # Training and validation loops
│   └── evaluate.py            # Metrics, confusion matrix, predictions
└── utils/
    └── seed.py                # Reproducibility utilities

scripts/
├── train.py                   # CLI entry point for training
└── extract_frames.py          # CLI entry point for frame extraction

experiments/
├── run_all.py                 # Run all experiments sequentially
├── learning_rate.py           # Learning rate sweep
├── dropout_rate.py            # Dropout rate comparison
├── batch_size.py              # Batch size comparison
├── augmentation.py            # Data augmentation strategies
├── freezing.py                # Frozen vs. fine-tuned backbone
└── optimizer.py               # AdamW vs. Adam vs. SGD
```

## Installation

```bash
git clone https://github.com/mehrdadmmz/RageVision.git
cd RageVision

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Dataset

The dataset consists of ~10,400 frames extracted from Twitch streaming videos, organized into two classes:

```
dataset/
├── rage/           # ~5,400 frames showing rage outbursts
│   ├── rage_0001.jpg
│   └── ...
└── non-rage/       # ~5,040 frames of normal content
    ├── non_rage_0001.jpg
    └── ...
```

**Data Augmentation (training):**
- Resize to 256x256 followed by RandomResizedCrop to 224x224
- Random rotation (±20°) and horizontal flip
- ImageNet normalization

To extract frames from your own videos:
```bash
python scripts/extract_frames.py \
    --input-dir path/to/videos \
    --output-dir dataset/rage \
    --prefix rage \
    --max-frames 90
```

## Usage

### Training

```bash
python scripts/train.py
```

With custom parameters:
```bash
python scripts/train.py \
    --epochs 15 \
    --lr 5e-4 \
    --data-dir /path/to/dataset \
    --save-path checkpoints/model.pth
```

### Running Experiments

Run all hyperparameter experiments:
```bash
python -m experiments.run_all
```

Or run individual experiments:
```bash
python -m experiments.learning_rate
python -m experiments.dropout_rate
python -m experiments.batch_size
python -m experiments.augmentation
python -m experiments.freezing
python -m experiments.optimizer
```

## Results

### Training and Validation Loss

The model shows steady convergence with decreasing loss over 10 epochs, indicating effective learning without significant overfitting.

<img width="709" alt="Training and Validation Loss" src="https://github.com/user-attachments/assets/b2b7b327-4477-4243-a277-fe3f3cf9539c" />

### Confusion Matrix

<img width="476" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/e09fc7b9-ee64-47d5-8b39-4ea863f1628f" />

| Metric | Non-Rage | Rage |
|---|---|---|
| Precision | 0.89 | 0.83 |
| Recall | 0.82 | 0.90 |
| F1-Score | 0.85 | 0.86 |
| **Overall Accuracy** | | **86%** |

**Confusion Matrix Breakdown:**
- 964 / 1,075 rage frames correctly classified (90% recall)
- 885 / 1,085 non-rage frames correctly classified (82% recall)
- 200 non-rage frames misclassified as rage (false positives)
- 111 rage frames misclassified as non-rage (false negatives)

The higher recall for the rage class (0.90) indicates the model is particularly good at catching genuine rage moments, which is the primary objective.

## Experiments

Six hyperparameter experiments were conducted to analyze model behavior:

| Experiment | Variants Tested | Best Config |
|---|---|---|
| Learning Rate | 1e-2, 5e-3, 1e-3, 7e-4, 5e-4, 1e-4, 1e-5 | 1e-3 |
| Dropout Rate | 0.1, 0.3, 0.5, 0.7, 0.9 | 0.3 |
| Batch Size | 8, 16, 32, 64, 128 | 32 |
| Augmentation | None, Moderate, Heavy | Moderate |
| Training Strategy | Frozen backbone, Full fine-tune | Frozen |
| Optimizer | AdamW, Adam, SGD | AdamW |

## Contributors

- [Mehrdad Momeni zadeh](https://github.com/mehrdadmmz) - mma236@sfu.ca
- [Zheng (Arthur) Li](https://github.com/Mercury-AL) - zla229@sfu.ca
- [Daniel Surina](https://github.com/sosokokos) - dsa108@sfu.ca
