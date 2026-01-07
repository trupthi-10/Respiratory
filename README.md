# Respiratory
This project presents a multimodal AI-based lung disease detection system that integrates physiological parameters such as heart rate, SpO₂ and blood pressure with chest X-ray analysis. Vitals data is used to assess lung health risk, while deep learning models analyze chest X-rays to accurately detect and classify the presence of lung diseases.

# Multimodal AI System for Lung Disease Detection

## Project Overview
This project implements a multimodal AI-based lung disease detection system that integrates physiological health parameters with chest X-ray image analysis to assess lung health risk and accurately identify specific lung diseases. By combining numerical vitals data and medical imaging, the system improves diagnostic accuracy and supports early disease detection.

## Objectives
- Assess lung health risk using physiological parameters  
- Detect and classify lung diseases from chest X-ray images  
- Combine multiple data modalities for reliable diagnosis  
- Provide confidence-based and explainable predictions  

## System Architecture
The system follows a two-stage pipeline:

### 1. Vitals-Based Lung Health Risk Prediction
Physiological parameters used:
- Heart Rate  
- SpO₂ (Oxygen Saturation)  
- Body Temperature  
- Blood Pressure  

These parameters are analyzed to predict overall lung health risk.

### 2. Chest X-ray Disease Detection
Chest X-ray images are processed using deep learning models to accurately detect and classify lung diseases.

## Technologies Used
- Programming Language: Python  
- Frameworks & Libraries: TensorFlow / PyTorch, OpenCV, NumPy, Pandas, Scikit-learn  
- Backend: Flask  
- Database: MySQL  
- Explainable AI: Grad-CAM  

## Key Features
- Multimodal data fusion (Vitals + X-ray images)  
- Lung health risk prediction using physiological data  
- Disease-specific detection from chest X-rays  
- Prediction confidence score  
- Explainable AI visualization (Grad-CAM)  
- User-friendly web interface  
- Downloadable diagnostic report  

## Input Data
### Physiological Parameters
- Heart Rate  
- SpO₂  
- Temperature  
- Blood Pressure  

### Medical Imaging
- Chest X-ray images (Frontal view)

## Output
- Lung health risk assessment  
- Detected lung disease (e.g., Pneumonia, Tuberculosis, COVID-19)  
- Prediction confidence score  
- Visual explanation highlighting affected lung regions  
