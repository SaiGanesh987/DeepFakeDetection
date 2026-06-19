# DeepFake Detection using MobileNetV2

This project detects whether a face image or video is **Real** or **Fake** using a fine-tuned MobileNetV2 model. It is developed using TensorFlow and demonstrates an end-to-end deep learning workflow including data preprocessing, transfer learning, model training, evaluation, and inference.

## Features

- MobileNetV2 Transfer Learning
- Fine-tuning of pretrained network
- Image Classification
- Video Classification
- TensorFlow Data Pipeline
- Early Stopping
- Model Checkpointing
- Learning Rate Scheduling
- Evaluation Metrics
- Confusion Matrix
- Classification Report
- Training History Visualization
- Ready for Grad-CAM Explainability
- Ready for Streamlit Deployment

## Tech Stack

- Python
- TensorFlow
- MobileNetV2
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Pandas

## Dataset

Real vs Fake Faces (10K)

- 7,000 Training Images
- 3,000 Validation Images

Classes:
- Real
- Fake

## Model

Backbone:
- MobileNetV2 (ImageNet Pretrained)

Classifier:
- Global Average Pooling
- Dropout
- Dense (ReLU)
- Dropout
- Sigmoid Output Layer

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

## Future Improvements

- Grad-CAM Explainability
- ROC Curve
- Precision-Recall Curve
- Misclassified Image Analysis
- Webcam Detection
- Streamlit Web Application
- Docker Deployment