from orchestrator.task_router import route_task
from agents.classification_agent import ClassificationAgent
from agents.segmentation_agent import SegmentationAgent
import mlflow


def main():
    print("\n🧠 Welcome to the LLM-Directed Agentic ML Pipeline\n")
    user_query = input("🗣️ Enter your question or request about the dataset: ")

    # LLM interprets user intent 
    decision = route_task(user_query)
    print(f"\n🤖 LLM decided this is a {decision.upper()} task.\n")

    # MLflow setup 
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("LLM-Directed Agentic Pipeline")

    # Trigger the appropriate agent 
    if decision == "classification":
        clf_agent = ClassificationAgent(dataset_name="ShapeNetPart")
        clf_results = clf_agent.run_all_models()
        best = max(clf_results, key=lambda m: clf_results[m]["accuracy"])
        print(f"Best classifier: {best} ({clf_results[best]['accuracy']:.3f})")

    elif decision == "segmentation":
        seg_agent = SegmentationAgent(dataset_name="ShapeNetPart")
        seg_results = seg_agent.run_all_models()  
        best = max(seg_results, key=lambda m: seg_results[m]["iou"])
        print(f"Best segmentation model: {best} ({seg_results[best]['iou']:.3f})")

    else:
        print("Could not determine task type. Please rephrase your query.")

    print("\nPipeline complete. View results with `mlflow ui`.\n")


if __name__ == "__main__":
    main()
