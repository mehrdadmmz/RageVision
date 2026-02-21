"""Experiment: Data augmentation strategy comparison."""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
from torchvision.models import MobileNet_V2_Weights

from ragevision.config import (
    DATASET_PATH, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    TRAINING_PORTION, DROPOUT_RATE, LEARNING_RATE, NUM_CLASSES, NUM_EPOCHS,
)
from ragevision.data import val_transforms
from ragevision.utils import set_seed
from ragevision.training import train_one_epoch, validate

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

AUGMENTATION_STRATEGIES = {
    "No_Aug": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    "Moderate_Aug": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    "Heavy_Aug": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
}


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    print("\n" + "=" * 50)
    print("Experiment: Data Augmentation Strategies")
    print("=" * 50)

    for aug_name, transform in AUGMENTATION_STRATEGIES.items():
        print(f"\nStrategy: {aug_name}")
        set_seed(42)

        dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
        train_size = int(TRAINING_PORTION * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        val_ds.dataset.transform = val_transforms

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        )

        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
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

        results[aug_name] = val_acc

    plt.figure(figsize=(8, 6))
    plt.bar(list(results.keys()), list(results.values()))
    plt.xlabel("Augmentation Strategy")
    plt.ylabel("Validation Accuracy")
    plt.title("Data Augmentation vs. Validation Accuracy")
    plt.tight_layout()
    plt.savefig("augmentation_experiment.png", dpi=150)
    plt.show()

    return results


if __name__ == "__main__":
    run()
