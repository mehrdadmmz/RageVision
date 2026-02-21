"""Train the RageVision rage classifier."""

import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from ragevision.config import NUM_EPOCHS, LEARNING_RATE, NUM_CLASSES
from ragevision.utils import set_seed
from ragevision.data import get_dataloaders
from ragevision.models import build_model
from ragevision.training import train_one_epoch, validate, get_predictions, plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Train the RageVision classifier")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to dataset directory")
    parser.add_argument("--save-path", type=str, default="best_model.pth", help="Path to save best model")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _, _ = get_dataloaders(dataset_path=args.data_dir)

    model = build_model(pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    start = datetime.now()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch + 1}/{args.epochs} [{elapsed:.1f}s]: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> Saved best model to {args.save_path}")

    total_time = datetime.now() - start
    print(f"\nTraining completed in {total_time}")

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, args.epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=150)
    plt.show()

    # Evaluate best model
    model.load_state_dict(torch.load(args.save_path, weights_only=True))
    preds, labels = get_predictions(model, val_loader, device)

    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(labels, preds))

    class_names = ["non-rage", "rage"]
    plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png")


if __name__ == "__main__":
    main()
