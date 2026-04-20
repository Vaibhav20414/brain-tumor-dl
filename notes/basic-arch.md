Architecture
Backbone — Custom CNN (src/models/backbone.py)
A 5-block convolutional feature extractor:


Input (3×224×224)
→ Conv2d(3→32) → BN → ReLU → MaxPool
→ Conv2d(32→64) → BN → ReLU → MaxPool
→ Conv2d(64→128) → BN → ReLU → MaxPool
→ Conv2d(128→256) → BN → ReLU → MaxPool
→ Conv2d(256→512) → BN → ReLU → MaxPool
→ Global Average Pooling
→ 512-d embedding
Batch Normalization after every conv layer stabilizes training and allows higher learning rates. Global Average Pooling replaces a large flatten+dense layer — fewer parameters, built-in spatial regularization.

Task Heads
DetectionHead: Linear(512→256) → ReLU → Dropout(0.4) → Linear(256→1) — single logit for BCE loss
ClassificationHead: Linear(512→256) → ReLU → Dropout(0.5) → Linear(256→4) — 4-class logits for CE loss
Two-Phase Transfer Learning
The key architectural decision: Phase 2 reuses Phase 1's backbone weights.


Phase 1 trains:  [Backbone] + [DetectionHead]
                      ↓ transfer weights
Phase 2 trains:  [Backbone] + [ClassificationHead]
The backbone learned general MRI feature representations (edges, textures, anatomical structures) during binary detection. Phase 2 fine-tunes these representations for fine-grained tumor type discrimination — requiring far fewer epochs to converge.

Training Techniques
Technique   	Phase 1	        Phase 2
Loss	    BCEWithLogitsLoss	CrossEntropyLoss + label smoothing (0.1)
Optimizer	Adam (lr=0.001)	Adam (lr=0.0005)
Scheduler	CosineAnnealingLR	CosineAnnealingLR
Early Stopping	patience=7	patience=10
Dropout	0.4	0.5
Label smoothing (0.1) in Phase 2 prevents the model from becoming overconfident — it softens one-hot targets from [0,0,1,0] to [0.025, 0.025, 0.925, 0.025], improving calibration and generalization.

Cosine Annealing decays learning rate smoothly from lr_max to 0 following a cosine curve — avoids the sharp drops of step-decay and helps escape local minima near the end of training.

Data Pipeline
Stratified 70/15/15 split using sklearn's StratifiedShuffleSplit — preserves class distribution across train/val/test even with imbalanced classes.

Augmentation (train only):

RandomHorizontalFlip — MRI symmetry is diagnostically valid
RandomRotation(±15°) — patient positioning variance
ColorJitter — scanner contrast variance
Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) — ImageNet stats, transfers well to medical grayscale converted to RGB
Val/test: resize + normalize only — no augmentation to ensure clean evaluation.

Results
Metric	Score
Phase 1 (test set)	F1 / AUC	0.857 / 0.881
Phase 2 (test set)	Macro F1	0.975
Phase 1 achieves 100% recall — it misses zero tumors (8 false positives, 0 false negatives). In a medical screening context, this is the right trade-off: false positives get reviewed by a clinician, false negatives go undetected.

Phase 2 trained to epoch 36/40 (val score 0.975), exceeding the ≥0.88 target by a large margin.

Serving
FastAPI inference server with:

POST /api/predict — runs both models sequentially, returns structured JSON with detection confidence, tumor type, and per-class probabilities
GPU inference on RTX 4050 via CUDA 12.4
Static SPA frontend with drag-and-drop upload and animated confidence ring visualization
Key Design Decisions
Phase-agnostic Trainer — the same training loop handles both tasks; task-specific logic lives only in the loss function and head
Config-driven — no hardcoded hyperparameters in source files; YAML configs are the single source of truth
Early stopping on val F1 (not val loss) — directly optimizes the metric that matters, not a proxy