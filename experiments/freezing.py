"""Experiment: Frozen feature extractor vs. full fine-tuning."""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

from ragevision.config import DROPOUT_RATE, LEARNING_RATE, NUM_CLASSES, NUM_EPOCHS
from ragevision.utils import set_seed
from ragevision.data import get_dataloaders
from ragevision.training import train_one_epoch, validate


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _, _ = get_dataloaders()

    strategies = {"Freeze": True, "FineTune": False}
    results = {}

    print("\n" + "=" * 50)
    print("Experiment: Freezing vs. Fine-Tuning")
    print("=" * 50)

    for name, freeze in strategies.items():
        print(f"\nStrategy: {name}")
        set_seed(42)

        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(model.last_channel, NUM_CLASSES),
        )
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        params = model.classifier.parameters() if freeze else model.parameters()
        optimizer = optim.Adam(params, lr=LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            print(f"  Epoch {epoch + 1}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        results[name] = val_acc

    plt.figure(figsize=(8, 6))
    plt.bar(list(results.keys()), list(results.values()))
    plt.xlabel("Training Strategy")
    plt.ylabel("Validation Accuracy")
    plt.title("Freezing vs. Fine-Tuning Comparison")
    plt.tight_layout()
    plt.savefig("freezing_experiment.png", dpi=150)
    plt.show()

    return results


if __name__ == "__main__":
    run()
