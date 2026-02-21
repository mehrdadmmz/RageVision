"""Data loading and preprocessing pipeline."""

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

from ragevision.config import (
    DATASET_PATH, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, TRAINING_PORTION,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def get_dataloaders(dataset_path=None, batch_size=None):
    """Create train and validation DataLoaders.

    Args:
        dataset_path: Path to the dataset directory. Defaults to config value.
        batch_size: Batch size for DataLoaders. Defaults to config value.

    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset).
    """
    dataset_path = dataset_path or DATASET_PATH
    batch_size = batch_size or BATCH_SIZE

    dataset = datasets.ImageFolder(dataset_path, transform=train_transforms)

    train_size = int(TRAINING_PORTION * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    val_dataset.transform = val_transforms

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, val_loader, train_dataset, val_dataset
