import mlflow
import torch
from torch import nn, optim
from utils.data_utils import get_pointcloud_dataloaders
from utils.segmentation_utils import mean_iou
from models.pointnet import PointNetSegmentation
from .base_agent import BaseAgent
from tqdm import tqdm

class SegmentationAgent(BaseAgent):
    def __init__(self, dataset_name="ShapeNetPart", num_parts=6):
        super().__init__(dataset_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PointNetSegmentation(num_parts)

    def run_all_models(self):
        train_loader, val_loader = get_pointcloud_dataloaders()
        results = {}
        with mlflow.start_run(run_name="PointCloud-Segmentation"):
            model = self.model.to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            for epoch in range(5):
                model.train()
                total_loss = 0
                for pts, lbl, seg in tqdm(train_loader, desc=f"Epoch {epoch+1}/5"):
                    pts, seg = pts.to(self.device), seg.to(self.device)
                    optimizer.zero_grad()
                    preds = model(pts)              # [B,N,num_parts]
                    preds = preds.view(-1, preds.size(-1))
                    seg = seg.view(-1)
                    loss = criterion(preds, seg)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

            # Evaluate
            model.eval()
            miou = []
            with torch.no_grad():
                for pts, lbl, seg in val_loader:
                    pts, seg = pts.to(self.device), seg.to(self.device)
                    out = model(pts).argmax(-1)
                    miou.append(mean_iou(out, seg, num_classes=model.mlp4[-1].out_channels))
            mean_iou_val = sum(miou) / len(miou)
            mlflow.log_params({"dataset": self.dataset_name, "model": "PointNetSeg"})
            mlflow.log_metric("mean_iou", mean_iou_val)
            mlflow.pytorch.log_model(model, artifact_path="model")
            print(f"Validation mIoU: {mean_iou_val:.3f}")
            results["PointNetSeg"] = {"iou": mean_iou_val}
        return results
