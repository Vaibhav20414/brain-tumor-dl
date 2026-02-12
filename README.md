# brain-tumor-dl
CNN analysis of Brain MRI scans to perform a screening aid in medical context.

brain-mri-cnn/
│
├── PROJECT_CHARTER.md
├── README.md
├── requirements.txt
│
├── data/                # ignored by git
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │
│   ├── models/
│   │   ├── backbone.py
│   │   ├── detection_head.py
│   │   ├── classification_head.py
│   │   └── multitask_model.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── loss.py
│   │   └── metrics.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── confusion_matrix.py
│   │   └── error_analysis.py
│   │
│   └── utils/
│       ├── config.py
│       └── logger.py
│
├── experiments/
│   ├── exp_01_baseline/
│   ├── exp_02_multitask/
│   └── results_summary.md
│
├── models/              # saved weights (ignored)
└── notes/
    ├── data_inspection.md
    └── modeling_strategy.md
