import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetClassifier(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):  # x: [B, N, 3]
        x = x.transpose(2, 1)
        x = self.feat(x)
        x = torch.max(x, 2, keepdim=False)[0]
        return self.fc(x)


class PointNetSegmentation(nn.Module):
    def __init__(self, num_parts=6):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.mlp3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.mlp4 = nn.Sequential(
            nn.Conv1d(1216, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, num_parts, 1)
        )

    def forward(self, x):  # [B,N,3]
        x = x.transpose(2, 1)
        x1 = self.mlp1(x)
        x2 = self.mlp2(x1)
        x3 = self.mlp3(x2)
        global_feat = torch.max(x3, 2, keepdim=True)[0]
        global_feat = global_feat.repeat(1, 1, x.size(2))
        concat = torch.cat([x1, x2, global_feat], 1)
        out = self.mlp4(concat)
        out = out.transpose(2, 1).contiguous()
        return out  # logits [B,N,num_parts]
