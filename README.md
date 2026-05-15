# Smart Agriculture Crop Disease Detection 🌱🦠

**Objective:** Classify plant diseases from leaf images and recommend treatments for healthy crop management.

---

## Phase 1 — Data & Classical ML

- **Exploratory Analysis:** Analyze leaf images and disease labels.  
- **Feature Extraction:** Compute color histograms and texture features.  
- **Dimensionality Reduction:** Apply PCA on image features for visualization.  
- **Classical ML Models:**  
  - K-Nearest Neighbors (KNN)  
  - Naïve Bayes  
- **Ensemble Methods:**  
  - Random Forest  
  - AdaBoost  
- **Model Optimization:**  
  - Cross-validation  
  - Hyperparameter grid search  

---

## Phase 2 — Deep Learning (CNN / Autoencoder / Transfer Learning)

- **Convolutional Neural Network (CNN):** Train on PlantVillage dataset.  
- **Autoencoder (AE):** Detect anomalies between healthy and diseased leaves.  
- **Transfer Learning:** Fine-tune EfficientNet or MobileNetV2.  
- **Data Augmentation & Optimizers:** Experiment with rotations, flips, brightness, and different optimizers.  
- **Evaluation:** Build a confusion matrix dashboard to monitor performance.  

---

## Phase 3 — Advanced (Transformers / GANs / Explainable AI)

- **Time-Series Analysis (LSTM):** Analyze crop health indicators over time.  
- **Attention Mechanism:** Focus on multi-scale leaf features for better detection.  
- **Generative Adversarial Networks (DCGAN):** Synthesize rare disease samples for data augmentation.  
- **Vision Transformer (ViT):** Classify leaf diseases with transformer architecture.  
- **Explainable AI (XAI):** Use LIME and Grad-CAM for visualizing and localizing disease regions.  

---

## Dataset

- **Primary Dataset:** [PlantVillage dataset] (Kaggle)  https://www.kaggle.com/datasets/emmarex/plantdisease
   
- **Optional:** Crop type satellite imagery for additional features.  

---

## Tools & Libraries

- Python, OpenCV, Scikit-learn, TensorFlow/Keras, PyTorch, Pandas, NumPy, Matplotlib/Seaborn
-  **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt 

---

## Notes

- This project progresses from **classical ML** to **deep learning**, and finally to **advanced AI techniques**.  
- Each phase builds on the previous one, improving performance and explainability.
