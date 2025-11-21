from torchvision import models
import torch.nn as nn

def build_efficientnet_b0(num_classes=10):
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
