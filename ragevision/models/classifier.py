"""MobileNetV2-based rage classifier."""

import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

from ragevision.config import NUM_CLASSES, DROPOUT_RATE


def build_model(pretrained=True, num_classes=None, dropout_rate=None):
    """Build a MobileNetV2 model with a custom classification head.

    The base MobileNetV2 feature extractor is frozen and only the
    classification head is trained, following a transfer-learning approach.

    Args:
        pretrained: Whether to load ImageNet-pretrained weights.
        num_classes: Number of output classes. Defaults to config value.
        dropout_rate: Dropout probability. Defaults to config value.

    Returns:
        A MobileNetV2 model with frozen backbone and trainable classifier.
    """
    num_classes = num_classes or NUM_CLASSES
    dropout_rate = dropout_rate if dropout_rate is not None else DROPOUT_RATE

    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(model.last_channel, num_classes),
    )

    return model
