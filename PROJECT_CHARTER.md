**Problem Definition**- 

1. Binary classification: Tumor vs No Tumor 
2. Analysis of MRI Brain scans.
3. Single-Image interence.

**Objective**-
Primary: Minimize False Negative Rate
Secondary: Report accuracy and precision for context

Dataset Commitment- 

Kaggle Brain Tumor MRI Dataset
Label collapsing decision - We are just trying to classify that the tumor is present or not.

**Assumption**-
*Data Assumptions*- 

1. Kaggle labels are reasonalbly accurate.
2. Single MRI slice is sufficient.
3. All tumor types are treated equally.

*Modeling assumptions*-

4. Visual patterns are learnable.
5. Training distributions ~ test distribution 

*Decision assumptions*-

6. High recall is more important than precision.
7. Model is a screening aid not a definitive diagnosis.

**Non-goals** 

What are not my goals with this project? 
1. Multi-class tumor classification.
2. Clinical deployment. 
3. 3D MRI volumes.
4. Segmentation.


**Ethical & Safety Statement** 

Project intentions lies within the constraint of screening aid, not to be used for any definitive medical diagnosis. Any confirming statement or clinical decision must be made by a qualified medical professional, using the provided tool as a probability metric for a single MRI image screening. This project trains the model under the assumption that the provided dataset labels is offers reasonably accurate on data, while acknowledging the possibility of label noise, dataset bias, and limited clinical representativeness. 


