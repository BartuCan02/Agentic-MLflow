import json
import os
from datetime import datetime

def save_inference_json(output_dict, output_dir="results"):
    """
    Saves inference output (predictions, metadata, etc.) as a JSON file.

    Args:
        output_dict (dict): dictionary with inference results and metadata.
        output_dir (str): directory where the JSON file will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"inference_result_{timestamp}.json"
    path = os.path.join(output_dir, filename)

    with open(path, "w") as f:
        json.dump(output_dict, f, indent=2)

    print(f"💾 Results saved to: {path}")
    return path
