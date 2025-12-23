import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointtransformer_v3 import PointTransformerV3Layer
from .base_agent import BaseAgent

class PointCloudClassifier(nn.Module):
    """A full classification model built from PointTransformerV3Layer."""
    def __init__(self, num_classes, model_dim=64):
        super().__init__()
        self.input_proj = nn.Linear(3, model_dim)
        
        self.transformer_layers = nn.ModuleList([
            PointTransformerV3Layer(dim=model_dim, num_neighbors=16),
            PointTransformerV3Layer(dim=model_dim, num_neighbors=16)
        ])
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_fc = nn.Linear(model_dim, num_classes)

    def forward(self, point_cloud):
        # Input shape: [B, N, 3]
        coords = point_cloud
        # features shape: [B, N, model_dim]
        features = self.input_proj(coords)
        
        # Manually iterate through layers
        # The layer expects (B, N, D) for features and coords
        for layer in self.transformer_layers:
            features = layer(features, pos=coords)
        
        # Pool features across all points. Pool expects (B, D, N)
        # so we permute before pooling.
        pooled_features = self.pool(features.permute(0, 2, 1)).squeeze(-1)
        
        # Classify
        logits = self.output_fc(pooled_features)
        return logits

class ClassificationAgent(BaseAgent):
    def __init__(self, model_path, num_classes=16):
        super().__init__("BridgeDataset")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PointCloudClassifier(num_classes=num_classes)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def run_inference(self, point_cloud):
        point_cloud = point_cloud.to(self.device)
        logits = self.model(point_cloud)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        return pred.item(), conf.item()

