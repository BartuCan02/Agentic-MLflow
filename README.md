# Agentic-MLflow
# 🧠 LLM-Directed Agentic ML Pipeline for Point-Cloud Classification & Segmentation

This project implements an **agentic AI system** that combines **Large Language Model (LLM) reasoning** with **autonomous machine-learning agents** for 3D point-cloud data.  
The pipeline can perform both **object classification** and **part segmentation** on the *same* dataset (e.g., ShapeNetPart), while automatically tracking experiments with **MLflow**.

---

## 🚀 Overview

### 🧩 Architecture

1. **User Input (Natural Language)**
   - Example:  
     - “What is this object? Is it a bridge?” → *Classification*  
     - “Segment the cars in this scene.” → *Segmentation*

2. **LLM Router**
   - Interprets the user query and decides whether to trigger the **Classification Agent** or the **Segmentation Agent**.

3. **Agentic ML Pipeline**
   - **ClassificationAgent** trains a PointNet model for category prediction.  
   - **SegmentationAgent** trains a PointNetSeg model for per-point part segmentation.

4. **MLflow Tracking**
   - Logs all parameters, metrics, and models.
   - Enables experiment comparison and performance visualization.

---



