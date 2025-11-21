import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class PointCloudDataset(Dataset):
    """
    Shared dataset for both classification and segmentation.
    Expects .npz files with keys 'points', 'label', 'seg_label'.
    Each sample: (N,3) points, single class label, and per-point part labels.
    """
    def __init__(self, root, split="train", num_points=2048):
        super().__init__()
        path = os.path.join(root, f"{split}.npz")
        data = np.load(path)
        self.points = data["points"]      # shape: (samples, N, 3)
        self.labels = data["labels"]      # shape: (samples,)
        self.seg_labels = data["seg_labels"]  # shape: (samples, N)
        self.num_points = num_points

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        pts = self.points[idx]
        lbl = self.labels[idx]
        seg = self.seg_labels[idx]

        # random sampling to fixed size
        choice = np.random.choice(len(pts), self.num_points, replace=True)
        pts = pts[choice, :]
        seg = seg[choice]

        pts = torch.tensor(pts, dtype=torch.float32)
        lbl = torch.tensor(lbl, dtype=torch.long)
        seg = torch.tensor(seg, dtype=torch.long)
        return pts, lbl, seg


def get_pointcloud_dataloaders(root="./data/shapenet", batch_size=16, num_points=2048):
    train_set = PointCloudDataset(root, "train", num_points)
    test_set = PointCloudDataset(root, "test", num_points)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
