🧠 Brain Tumor Detection using Deep Learning

CNN-based analysis of Brain MRI scans to assist medical screening through classification and multi-task learning.

📌 Project Overview

This project implements a modular deep learning pipeline for detecting brain tumors from MRI scans.

The system supports:

Binary / multi-class tumor classification

Multi-task learning (shared backbone + task-specific heads)

Experiment tracking

Structured evaluation with confusion matrix & error analysis

The goal is to simulate a production-style ML project structure, not just a notebook experiment.



🧠 Model Design
1️⃣ Backbone

CNN-based feature extractor

Shared representation learning

Easily replaceable (ResNet / EfficientNet ready)

2️⃣ Heads

Classification Head → Tumor prediction

Detection Head → (Optional) localization

Multi-task setup improves representation learning

⚙️ Training Pipeline

PyTorch Dataset abstraction

Modular loss functions

Metrics tracking (Accuracy, Precision, Recall, F1)

Confusion matrix generation

Error case inspection

📊 Evaluation Strategy

Evaluation includes:

Classification metrics

Confusion matrix visualization

Error case breakdown

Per-class performance analysis

This mirrors real-world ML validation workflows.