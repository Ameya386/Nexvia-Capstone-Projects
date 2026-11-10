import torch.nn as nn
from torchvision import models
from src import config as C


def build_model(pretrained=True):
    """DenseNet-121 (ImageNet transfer learning) with a 14-way multi-label head."""
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    m = models.densenet121(weights=weights)
    m.classifier = nn.Linear(m.classifier.in_features, len(C.LABELS))
    return m
