import mlflow
import torch
from torch import nn, optim
from utils.data_utils import get_pointcloud_dataloaders
from utils.train_eval_utils import train_model, evaluate_model
from models.pointnet import PointNetClassifier
from .base_agent import BaseAgent

class ClassificationAgent(BaseAgent):
    def __init__(self, dataset_name="ShapeNetPart", num_classes=16):
        super().__init__(dataset_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PointNetClassifier(num_classes)

    def run_all_models(self):
        train_loader, val_loader = get_pointcloud_dataloaders()
        results = {}
        with mlflow.start_run(run_name="PointCloud-Classification"):
            model = self.model.to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            train_model(model, train_loader, criterion, optimizer, self.device, epochs=5)
            acc = evaluate_model(model, val_loader, self.device)
            mlflow.log_params({"dataset": self.dataset_name, "augmentation": self.augmentation, "model": "PointNet"})
            mlflow.log_metric("accuracy", acc)
            mlflow.pytorch.log_model(model, artifact_path="model")
            results["PointNet"] = {"accuracy": acc}
        return results
