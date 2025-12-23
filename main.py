from orchestar.task_router import create_bridge_requirement_json
from agents.classification_agent import ClassificationAgent
from agents.segmentation_agent import SegmentationAgent
from utils.json_output_utils import save_inference_json

import torch
import numpy as np
import json
import os
import glob

def main():
    print("\n🧠 LLM-Directed Agentic ML Pipeline (Inference-Only)\n")

    # Step 1 – User query → LLM generates structured JSON
    user_query = input("🗣️ Enter your question about the 3D object: ")
    requirement_json = create_bridge_requirement_json(user_query)
    print("\n Generated requirement JSON:\n", json.dumps(requirement_json, indent=2))

    decision = requirement_json.get("task", "unknown")
    print(f"\n🤖 LLM decided this is a {decision.upper()} task.\n")

    # Step 2 – Load and downsample point cloud
    point_cloud_path = (
        "/Users/bartu/Desktop/Bartu/Python Projects/Hiwi_Computer Vision_LLM/"
        "agentic_pipeline/Semantic_Segmentation/Main/dataset/bridge_1/bridge_1/bridge_1.txt"
    )

    print(f" Loading point cloud from: {point_cloud_path}")
    full_point_cloud = np.loadtxt(point_cloud_path)
    print(f"✅ Full point cloud has {len(full_point_cloud)} points.")

    num_points_to_sample = 1024
    if len(full_point_cloud) > num_points_to_sample:
        print(f" Downsampling to {num_points_to_sample} points...")
        idx = np.random.choice(len(full_point_cloud), num_points_to_sample, replace=False)
        point_cloud = full_point_cloud[idx, :]
    else:
        point_cloud = full_point_cloud

    print(f"✅ Using {len(point_cloud)} points for inference.")
    point_cloud = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0)  # [1,N,3]


    # === CONFIGURABLE PATHS ===
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "Semantic_Segmentation", "Main", "dataset")
    BRIDGE_NAME = "bridge_3"  
    BRIDGE_SUBDIR = os.path.join(DATASET_DIR, BRIDGE_NAME, BRIDGE_NAME)
    POINT_CLOUD_FILE = os.path.join(BRIDGE_SUBDIR, f"{BRIDGE_NAME}.txt")
    ANNOTATION_DIR = os.path.join(DATASET_DIR, BRIDGE_NAME, BRIDGE_NAME, "Annotations")  

    # === CLASSIFICATION TASK ===
    if decision == "classification":
        point_cloud_path = POINT_CLOUD_FILE
        print(f"🏷️ Predicted class → {label} (confidence {conf:.2f})")

        # Save JSON output
        output_data = {
            "task": "classification",
            "model": {
                "name": "PointTransformerV3",
                "type": "Classification",
            },
            "input_file": point_cloud_path,
            "num_points": int(point_cloud.shape[1]),
            "predicted_class": int(label),
            "confidence": float(conf),
        }
        save_inference_json(output_data)

    # === SEGMENTATION TASK ===
    elif decision == "segmentation":
        # Use RandLANet as in SegmentationAgent
        print(f"Using model: RandLANet (Segmentation) via SegmentationAgent")
        seg_agent = SegmentationAgent()
        seg_map = seg_agent.run_inference(point_cloud)
        unique_segments = np.unique(seg_map)
        num_unique_segments = len(unique_segments)

        # Get segment names from annotation folder if not in requirement_json
        class_names = requirement_json.get("class_names")
        if not class_names:
            ann_dir = os.path.join(os.path.dirname(point_cloud_path), "Annotations")
            if os.path.isdir(ann_dir):
                class_names = sorted([f.split("_")[0] for f in os.listdir(ann_dir) if f.endswith(".txt")])
                requirement_json["class_names"] = class_names
        if class_names and isinstance(class_names, list):
            segment_names = [class_names[int(label)] if int(label) < len(class_names) else str(label) for label in unique_segments]
        else:
            segment_names = [str(label) for label in unique_segments]

        # Only print summary and save metrics to JSON
        annotation_dir = os.path.join(DATASET_DIR, BRIDGE_NAME, BRIDGE_NAME, "Annotations")
        annotation_files = sorted(glob.glob(os.path.join(annotation_dir, "*.txt")))
        pred_points = point_cloud.squeeze(0).cpu().numpy()[:, :3]  # [N, 3]
        pred_labels = seg_map  # [N]
        gt_masks = []
        class_names_eval = []
        iou_results = {}
        acc_results = {}
        for class_id, ann_file in enumerate(annotation_files):
            class_name = os.path.splitext(os.path.basename(ann_file))[0].replace("_3", "")
            class_names_eval.append(class_name)
            ann_data = np.loadtxt(ann_file)
            ann_points = ann_data[:, :3]
            ann_mask = ann_data[:, -1]  # last column, float
            mask = np.zeros(pred_points.shape[0], dtype=np.int32)
            for i, p in enumerate(pred_points):
                matches = np.where(np.all(np.isclose(ann_points, p, atol=1e-5), axis=1))[0]
                if len(matches) > 0:
                    gt_val = ann_mask[matches[0]]
                    mask[i] = int(gt_val > 0.5)
            gt_masks.append(mask)

        print("\nSegmentation Evaluation Results:")
        for class_id, (class_name, gt_mask) in enumerate(zip(class_names_eval, gt_masks)):
            pred_mask = (pred_labels == class_id).astype(np.int32)
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            iou = intersection / union if union > 0 else float('nan')
            acc = (pred_mask == gt_mask).mean()
            iou_results[class_name] = float(iou) if not np.isnan(iou) else None
            acc_results[class_name] = float(acc)
            print(f"  {class_name}: IoU={iou:.4f}, Accuracy={acc:.4f}")
        # Save JSON output with metrics
        output_data = {
            "task": "segmentation",
            "model": {
                "name": "RandLANet",
                "type": "Segmentation",
                "weights_path": "see SegmentationAgent"
            },
            "input_file": point_cloud_path,
            "num_points": int(point_cloud.shape[1]),
            "num_segments": int(num_unique_segments),
            "unique_segments": segment_names,
            "requirement_json": requirement_json,
            "metrics": {
                "iou": iou_results,
                "accuracy": acc_results
            }
        }
        save_inference_json(output_data)

    # === BOTH TASKS ===
    elif decision == "both":
        print("🧠 Using model: PointTransformerV3 (Classification & Segmentation)")

        # 1. Classification
        clf_path = "checkpoints/ptv3_classification.pth"
        clf_agent = ClassificationAgent(model_path=clf_path)
        label, conf = clf_agent.run_inference(point_cloud)
        print(f"🏷️ Predicted class → {label} (confidence {conf:.2f})")

        # 2. Segmentation
        seg_path = "checkpoints/ptv3_segmentation.pth"
        seg_agent = SegmentationAgent(model_path=seg_path)
        seg_map = seg_agent.run_inference(point_cloud)
        unique_segments = np.unique(seg_map)
        num_unique_segments = len(unique_segments)
        print(f"🧩 Found {num_unique_segments} unique segments: {unique_segments.tolist()}")

        # Save combined JSON
        output_data = {
            "task": "both",
            "models": {
                "classification": {"name": "PointTransformerV3", "weights_path": clf_path},
                "segmentation": {"name": "PointTransformerV3", "weights_path": seg_path}
            },
            "input_file": point_cloud_path,
            "num_points": int(point_cloud.shape[1]),
            "predicted_class": int(label),
            "confidence": float(conf),
            "num_segments": int(num_unique_segments),
            "unique_segments": unique_segments.tolist(),
            "requirement_json": requirement_json
        }
        save_inference_json(output_data)

    # === UNKNOWN TASK ===
    else:
        print(" Could not determine task type. Please rephrase your query.")

    print("\n Inference complete – View results with your preferred JSON viewer.\n")


if __name__ == "__main__":
    main()
