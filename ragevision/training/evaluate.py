"""Model evaluation and visualization utilities."""

import numpy as np
import torch
import matplotlib.pyplot as plt


def get_predictions(model, loader, device):
    """Collect predictions and ground-truth labels from a DataLoader.

    Args:
        model: The trained model.
        loader: DataLoader to run inference on.
        device: Torch device.

    Returns:
        Tuple of (predictions, labels) as numpy arrays.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm, classes, save_path=None):
    """Plot a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array.
        classes: List of class names.
        save_path: If provided, save the figure to this path.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
