
from .base_agent import BaseAgent
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
import os
import numpy as np
import torch
import open3d as o3d

class SegmentationAgent(BaseAgent):
    def __init__(self, model_path=None, num_parts=6):
        super().__init__("RandLANetBridge")
        # Set up paths
        ckpt_path = "/Users/bartu/Desktop/Bartu/Python Projects/Hiwi_Computer Vision_LLM/agentic_pipeline/Semantic_Segmentation/Results (leave one out cross validation (LOOCV))/bridge_3/logs/RandLANet_SCMSB_torch/checkpoint/ckpt_00280.pth"
        cfg_path = "/Users/bartu/Desktop/Bartu/Python Projects/Hiwi_Computer Vision_LLM/agentic_pipeline/Semantic_Segmentation/Main/randlanet_cms_bridge.yml"

        # Load config and model
        cfg = _ml3d.utils.Config.load_from_file(cfg_path)
        # Use Custom3D dataset for inference on a single file
        pointcloud_file = "/Users/bartu/Desktop/Bartu/Python Projects/Hiwi_Computer Vision_LLM/agentic_pipeline/Semantic_Segmentation/Main/dataset/bridge_1/bridge_1/bridge_1.txt"
        pointcloud_dir = os.path.dirname(pointcloud_file)
        custom_dataset_cfg = {
            "dataset_path": pointcloud_dir,
            "train_dir": pointcloud_dir,
            "test_dir": pointcloud_dir,
            "val_dir": pointcloud_dir,
            "name": "Custom3D",
            "class_names": cfg.dataset.get("class_names", ["part1", "part2", "part3", "part4", "part5", "part6"]),
            "num_classes": cfg.dataset.get("num_classes", 6)
        }
        self.dataset = ml3d.datasets.Custom3D(**custom_dataset_cfg)
        # Ensure in_channels is set to 10 to match checkpoint
        model_cfg = dict(cfg.model)
        model_cfg["in_channels"] = 10
        self.model = ml3d.models.RandLANet(**model_cfg)
        device = "cuda" if o3d.core.cuda.is_available() else "cpu"
        self.pipeline = ml3d.pipelines.SemanticSegmentation(model=self.model, dataset=self.dataset, device=device, **cfg.pipeline)
        self.pipeline.load_ckpt(ckpt_path)

    def run_inference(self, point_cloud):
        # point_cloud: torch.Tensor [1, N, 10] or numpy [N, 10]
        # Convert to numpy if needed
        if isinstance(point_cloud, torch.Tensor):
            pc_np = point_cloud.squeeze(0).cpu().numpy()
        else:
            pc_np = point_cloud
        # Split into xyz and features
        xyz = pc_np[:, :3]
        features = pc_np[:, 3:]
        labels = np.zeros(xyz.shape[0], dtype=np.int64)
        data = {"point": xyz, "feat": features, "label": labels}
        # Run inference
        result = self.pipeline.run_inference(data)
        print("Inference result:", result)
        # result['predict'] is the segmentation mask
        seg = result["predict_labels"]
        return seg

    def load_annotations(self, annotation_file):
        """
        Loads ground truth labels from an annotation file.
        Assumes one label per line, integer encoded.
        """
        labels = []
        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(int(line))
        return np.array(labels, dtype=np.int64)

    def evaluate_predictions(self, predictions, ground_truth):
        """
        Evaluates predictions against ground truth labels.
        Returns accuracy and per-class IoU.
        """
        assert len(predictions) == len(ground_truth), "Prediction and ground truth lengths do not match."
        accuracy = np.mean(predictions == ground_truth)

        # Compute per-class IoU
        num_classes = max(np.max(predictions), np.max(ground_truth)) + 1
        ious = []
        for cls in range(num_classes):
            pred_mask = predictions == cls
            gt_mask = ground_truth == cls
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            if union == 0:
                iou = float('nan')
            else:
                iou = intersection / union
            ious.append(iou)
        return {"accuracy": accuracy, "ious": ious}

