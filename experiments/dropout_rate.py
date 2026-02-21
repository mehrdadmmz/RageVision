"""Experiment: Dropout rate comparison."""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

from ragevision.config import LEARNING_RATE, NUM_CLASSES, NUM_EPOCHS
from ragevision.utils import set_seed
from ragevision.data import get_dataloaders
from ragevision.training import train_one_epoch, validate

DROPOUT_RATES = [0.1, 0.3, 0.5, 0.7, 0.9]


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _, _ = get_dataloaders()
    results = {}

    print("\n" + "=" * 50)
    print("Experiment: Dropout Rate Comparison")
    print("=" * 50)

    for dr in DROPOUT_RATES:
        print(f"\nDropout = {dr}")
        set_seed(42)

        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Dropout(dr),
            nn.Linear(model.last_channel, NUM_CLASSES),
        )
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            print(f"  Epoch {epoch + 1}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        results[dr] = val_acc

    plt.figure(figsize=(8, 6))
    plt.bar([str(dr) for dr in DROPOUT_RATES], list(results.values()))
    plt.xlabel("Dropout Rate")
    plt.ylabel("Validation Accuracy")
    plt.title("Dropout Rate vs. Validation Accuracy")
    plt.tight_layout()
    plt.savefig("dropout_experiment.png", dpi=150)
    plt.show()

    return results


if __name__ == "__main__":
    run()
