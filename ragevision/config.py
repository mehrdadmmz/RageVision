"""Hyperparameter configuration for RageVision."""

import os

# Dataset
DATASET_PATH = os.environ.get("RAGEVISION_DATA_DIR", "./dataset")

# DataLoader
BATCH_SIZE = 32
NUM_WORKERS = 8
PIN_MEMORY = True

# Model
NUM_CLASSES = 2
DROPOUT_RATE = 0.3

# Training
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
TRAINING_PORTION = 0.8
