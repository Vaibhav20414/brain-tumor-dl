“The dataset provides predefined training and testing splits, which are used as-is. However, since the dataset consists of multiple MRI slices per individual and patient identifiers are not available, there is a potential risk of slice-level data leakage. This limitation is acknowledged and reflected in the interpretation of results.”

📊 Data Inspection Summary
Kaggle Brain Tumor MRI Dataset vs BraTS Dataset
1️⃣ Kaggle Brain Tumor MRI Dataset
Dataset Nature
2D MRI slices
Organized into folders:

glioma
meningioma
pituitary
notumor
Pre-split into training and testing directories
Multiple slices per individual (slice-based dataset)

Observations from Inspection
🔹 Slice-Based Structure

Each patient appears to have:
10–12 MRI slices
Adjacent slices show gradual anatomical changes
High inter-slice correlation

Implication:
Images are not independent samples. There is potential slice-level data leakage if train/test splits are not patient-aware.

🔹 Label Granularity

Labels are image-level (no masks)

No bounding boxes
No segmentation maps
No anatomical landmarks
This restricts tasks to:

Image classification (binary or multi-class)
It does not support:

Segmentation

Precise spatial measurement

Tumor-to-structure distance computation
🔹 Modality Information

MRI type not clearly specified (T1, T2, FLAIR unknown)
Likely heterogeneous sources
This limits:

Clinical realism
Controlled modeling assumptions
🔹 Visual Characteristics
Tumors often appear as:
Asymmetric masses
Hyperintense regions
Distorted surrounding tissue
Some cases are subtle and not easily distinguishable by a non-expert

Strengths of Kaggle Dataset

Easy to use
Clean folder structure
Suitable for:
Binary detection
Multi-class classification
Good for beginner-level DL experiments
Limitations

No patient IDs
Slice-level correlation
No segmentation labels
No 3D volume structure
Limited clinical metadata

2️⃣ BraTS Dataset (Brain Tumor Segmentation Challenge)
Dataset Nature
3D MRI volumes
Multi-modal:
T1
T1c (contrast-enhanced)
T2
FLAIR

Provides:

Pixel-wise segmentation masks
Tumor subregion annotations
This is a research-grade dataset.
Observations from Inspection
🔹 Volumetric Structure
Each case includes:

Full 3D MRI volume
Multiple modalities per patient
Consistent spatial alignment
This enables:

True anatomical reasoning
Tumor localization
Distance measurements
🔹 Rich Annotations

Segmentation masks include:
Enhancing tumor
Tumor core
Edema

This supports:

Segmentation tasks
Tumor volume estimation
Centroid calculations
Distance-to-structure analysis (if hippocampus is available)

🔹 Patient-Level Splits

BraTS is structured per patient.
This avoids:

Slice-level leakage
Artificial metric inflation
Strengths of BraTS
Research-standard dataset

Supports:

Segmentation
3D CNNs
Spatial analysis
Clinical-style modeling

Suitable for:

Graduate-level research
Advanced projects
Limitations
Higher computational requirements
Complex preprocessing

Requires volumetric modeling knowledge

Not beginner-friendly
