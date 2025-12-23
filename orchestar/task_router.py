import json
from utils.llm_utils import ask_llm


def create_bridge_requirement_json(user_query: str):
    """
    Ask an LLM to generate a structured JSON describing the user's query.

    Expected JSON keys:
      - task: "classification" | "segmentation" | "both"
      - target: e.g. "bridge"
      - properties: dictionary of requested features (segments, measurements, etc.)
    """
    system_prompt = (
        "You are an AI that converts a user's question about a bridge "
        "into a structured JSON specification. "
        "Use the keys: 'task', 'target', and 'properties'. "
        "The 'task' should be one of ['classification', 'segmentation', 'both']. "
        "If the query mentions bridge parts or geometry (e.g. height, width, deck), "
        "list them under 'properties'. Output ONLY valid JSON."
    )

    llm_response = ask_llm(system_prompt, user_query)

    # Try to parse JSON
    try:
        requirement_json = json.loads(llm_response)
    except Exception:
        # Fallback rule-based interpretation
        is_classification_query = any(k in user_query.lower() for k in ["what is", "is it", "identify", "class"])
        is_segmentation_query = any(k in user_query.lower() for k in ["segment", "separate", "parts"])

        if is_classification_query and is_segmentation_query:
            task = "both"
        elif is_segmentation_query:
            task = "segmentation"
        elif is_classification_query:
            task = "classification"
        else:
            task = "unknown"

        requirement_json = {
            "task": task,
            "target": "bridge",
            "properties": {}
        }

    return requirement_json
