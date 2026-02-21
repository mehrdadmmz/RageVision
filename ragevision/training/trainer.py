"""Training and validation loops."""

import torch
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run a single training epoch.

    Args:
        model: The neural network model.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer instance.
        device: Torch device (cuda/cpu/mps).

    Returns:
        Tuple of (epoch_loss, epoch_accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Evaluate the model on a validation set.

    Args:
        model: The neural network model.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Torch device (cuda/cpu/mps).

    Returns:
        Tuple of (epoch_loss, epoch_accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc
