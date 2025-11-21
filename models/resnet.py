from torchvision import models
import torch.nn as nn

def build_resnet18(num_classes=10):
    model = models.resnet18(weights="IMAGENET1K_V1")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
