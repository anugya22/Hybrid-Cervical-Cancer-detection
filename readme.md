# 🧬 Hybrid Cervical Cancer Risk Prediction

## 📌 Project Overview
This project implements a **hybrid deep learning model** for predicting cervical cancer risk, combining **visual features** from medical images and **tabular clinical/symptom data**.  
The motivation is that cancer diagnosis is not solely dependent on imaging or symptoms but benefits from a **multi-modal approach**.

---

## 🧠 Approach

### 🔹 1. Visual Features
- A **pretrained Convolutional Neural Network (CNN)** (EfficientNetB0–B1, transfer learning from ImageNet weights) was used to extract deep visual features from cervical cell images.
- Final layers were **fine-tuned** for our dataset while keeping lower convolutional blocks frozen to reduce overfitting.
- **Grad-CAM** was used for explainability, highlighting image regions most relevant to the prediction.

### 🔹 2. Tabular Features
- Clinical and symptom metadata were fed into a **Multi-Layer Perceptron (MLP)**.
- **Input preprocessing**:
  - Missing values imputed.
  - Features normalized using `StandardScaler`.
- **Architecture**:
  - 2–3 dense layers
  - BatchNorm + ReLU activation
  - Dropout for regularization

### 🔹 3. Hybrid Fusion
- **Late fusion strategy** was applied:
  - CNN image embeddings (e.g., 512-dim) + Tabular embeddings (e.g., 64-dim)  
  - Concatenated into a joint representation
  - Passed through a fully connected classifier for final prediction
- This allows the model to **leverage both modalities effectively**.

---

## ⚙️ Implementation Highlights
- **Framework**: TensorFlow/Keras (`Model` API)
- **Optimizer**: Adam (`lr=1e-4`)
- **Loss**: Binary Crossentropy (risk classification)
- **Regularization**:
  - Dropout layers in both CNN and MLP branches
  - Data augmentation on images (rotation, flips, zoom)
  - Early stopping on validation accuracy
- **Training data**: Two datasets combined to increase diversity and reduce overfitting
- **Model saving**:
  ```python
  model.save("hybrid_cervical_model.h5")
📊 Results
Training Performance
Accuracy: 97.87%

Loss: 0.0277

Evaluation Metrics
Precision: 0.8402

Recall: 0.9167

F1-Score: 0.8768

✅ These metrics indicate that the hybrid approach successfully balances sensitivity (recall) and specificity (precision).
## 🚧 Challenges Faced: Overfitting

During the initial training, the model showed signs of **severe overfitting**:

- Training accuracy rapidly climbed to ~100%.  
- Validation accuracy plateaued around ~92%.  
- Loss curves showed a large gap between training and validation.  

<p align="center">
 <img width="868" height="602" alt="Screenshot 2025-06-23 182543" src="https://github.com/user-attachments/assets/d4deedff-465f-4ff5-9c2b-55d17477d414" />

</p>

This indicated that the model was **memorizing training samples** instead of learning generalizable patterns.

---

## ✅ Fixes Applied

To address overfitting, I made several improvements:

- **Data Augmentation**: Added rotations, flips, zoom, brightness/contrast shifts.  
- **Regularization**: Applied Dropout (0.2–0.3) in both CNN and MLP branches.  
- **Early Stopping**: Stopped training when validation accuracy stopped improving.  
- **Balanced Dataset**: Verified class distribution to avoid class bias.  

---

## 📈 Improved Results

After applying these fixes, the training behavior improved:

- Training and validation accuracies are closer.  
- Validation loss no longer diverges dramatically.  
- Model generalizes better across all five cervical cell classes.  

<p align="center">
 <img width="938" height="441" alt="Screenshot 2025-09-14 220613" src="https://github.com/user-attachments/assets/526c0674-1350-404c-a477-14bd30abdad8" />


</p>

Additionally, I confirmed the dataset was **balanced across all classes**:

<p align="center">
<img width="434" height="309" alt="image" src="https://github.com/user-attachments/assets/a2a5c12e-ee89-4700-b432-962562f7c1f6" />

</p>

---

## 📊 Final Metrics (Hybrid Model)

- **Accuracy**: 97.8%  
- **Precision**: 0.84  
- **Recall**: 0.91  
- **F1-score**: 0.87  

I also applied **Grad-CAM** to visualize which regions the CNN focused on, confirming the model’s decisions are interpretable.

🔎 Explainability with Grad-CAM
Grad-CAM was applied to the CNN branch to visualize model attention.

Generated heatmaps highlight cell regions influencing predictions.

The tabular features strengthened hybrid prediction confidence.

Builds trust and transparency in the model’s predictions.

<img width="664" height="314" alt="image" src="https://github.com/user-attachments/assets/865dce62-7b27-4c43-9508-c95e055fc751" />
<img width="694" height="264" alt="image" src="https://github.com/user-attachments/assets/951ae0dc-3924-49e6-a6ff-b9a3671df7ad" />
<img width="1271" height="609" alt="Screenshot 2025-09-14 220337" src="https://github.com/user-attachments/assets/6eea103e-f58f-481d-b8cc-5824c81c5b9a" />


📂 Repository Structure
bash
Copy code
/ (project root)
├─ data/                # (Datasets will be linked here)        
├─ src/
│  ├─ models/           # Image CNN (EfficientNet), Tabular MLP, Fusion model
│  ├─ train.py          # Training pipeline
│  ├─ inference.py      # Inference pipeline
│  └─ explain.py        # Grad-CAM visualization
├─ outputs/
│  ├─ checkpoints/      # Saved weights
│  ├─ figures/          # Grad-CAM heatmaps, plots
├─ requirements.txt
└─ README.md

📈 Key Contributions
Designed a multi-modal hybrid model combining CNN + MLP for cervical cancer risk prediction.

Achieved >97% accuracy with a strong F1 score (0.87).

Added Grad-CAM explainability for medical interpretability.

Used two datasets to reduce bias and improve generalization.

Balanced performance using data augmentation and dropout regularization.

🔗 Datasets
Image dataset: https://www.kaggle.com/datasets/marinaeplissiti/sipakmed
Tabular data: https://www.kaggle.com/datasets/loveall/cervical-cancer-risk-classification


🚀 Future Work
Extend to larger, multi-institutional datasets

Experiment with attention mechanisms in the fusion layer

Deploy as a Flask/Streamlit web app for clinical usability
