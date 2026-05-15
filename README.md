# 🌱 Smart Agriculture Crop Disease Detection System

An AI-powered Smart Agriculture Crop Disease Detection System designed to identify and classify plant diseases using Deep Learning, Computer Vision, and Transfer Learning techniques. The system helps farmers and agricultural researchers detect crop diseases in real time and receive treatment recommendations through an interactive web application.

---

## Features

* Real-time plant disease prediction
* Automatic crop disease classification
* Upload plant leaf images for instant analysis
* Confidence score visualization
* Disease treatment recommendations
* Grad-CAM visualization for model interpretability
* Comparison of multiple Deep Learning architectures
* Interactive Gradio-based web interface
* High-accuracy CNN and Transfer Learning models

---

## Dataset

The project uses the **PlantVillage Dataset**, containing:

* 54,000+ labeled plant leaf images
* 15 disease classes
* Healthy and infected crop categories
* Multiple crop species

Dataset includes various crop diseases such as:

* Tomato Early Blight
* Tomato Late Blight
* Potato Early Blight
* Pepper Bell Bacterial Spot
* Healthy Leaf Classes
* And more...

---

## Models Implemented

This project includes implementation and comparison of multiple Machine Learning and Deep Learning models:

### Classical Machine Learning

* Support Vector Machine (SVM)
* Random Forest
* K-Nearest Neighbors (KNN)

### Deep Learning Models

* Custom CNN
* VGG-style CNN
* MobileNetV2
* EfficientNetB0
* Vision Transformer (ViT)
* Attention-based CNN with CBAM
* LSTM-based architectures

---

## 📊 Model Performance

| Model                    | Validation Accuracy |
| ------------------------ | ------------------- |
| Custom VGG-style CNN     | ~97.5%              |
| MobileNetV2              | ~93.6%              |
| Vision Transformer (ViT) | ~93.3%              |
| LSTM                     | ~91.7%              |

The custom-designed 4-block VGG-style CNN achieved the best overall performance.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* PyTorch
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Gradio
* CNN
* Transfer Learning
* Vision Transformer (ViT)
* CBAM Attention Module
* Grad-CAM
* Deep Learning
* Computer Vision
* Git & GitHub

---

## System Architecture

1. Image Upload
2. Image Preprocessing
3. Data Augmentation
4. Feature Extraction
5. Disease Classification
6. Confidence Score Generation
7. Treatment Recommendation
8. Real-time Prediction Interface

---

## Image Processing Pipeline

The system applies several preprocessing techniques:

* Image resizing
* Normalization
* Data augmentation
* Noise reduction
* Feature extraction
* Batch processing

---

## Advanced Features

### Transfer Learning

Pretrained models such as MobileNetV2 and EfficientNetB0 were fine-tuned for improved classification performance.

### Vision Transformer (ViT)

Implemented Transformer-based image classification using patch embeddings and self-attention mechanisms.

### Attention Mechanism (CBAM)

Integrated Convolutional Block Attention Module (CBAM) to improve feature learning and focus on disease regions.

### Grad-CAM Visualization

Used Grad-CAM for visualizing disease regions and improving model interpretability.

---

## 💻 Installation

### Clone Repository

```bash
git clone https://github.com/hagerah2005/Smart-Agriculture-Crop-Disease-Detection.git
cd Smart-Agriculture-Crop-Disease-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

---

## Screenshots

### Disease Prediction Interface

(Add screenshot here)

### Model Prediction Results

(Add screenshot here)

---

## Future Improvements

* Mobile application deployment
* Cloud-based AI prediction API
* Additional crop disease classes
* IoT sensor integration
* Real-time drone monitoring
* Multi-language support
* Edge AI optimization

---

## Applications

* Smart Farming
* Precision Agriculture
* Crop Health Monitoring
* Agricultural Research
* Automated Disease Detection
* AI-based Farming Assistance

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---


