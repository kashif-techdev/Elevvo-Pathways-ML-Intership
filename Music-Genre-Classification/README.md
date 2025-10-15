# 🎵 Music Genre Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![Librosa](https://img.shields.io/badge/Librosa-0.10+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)

**AI-Powered Music Genre Classification System**

*Classify music genres using both feature-based and image-based approaches*

[🚀 Quick Start](#-quick-start) • [📊 Features](#-features) • [🎯 Demo](#-demo) • [📚 Documentation](#-documentation)

</div>

---

A comprehensive music genre classification system that implements **two different approaches** for classifying music into 10 different genres using the GTZAN dataset.

## 🚀 Features

### **Dual Approach Implementation**
- **Feature-Based Approach**: Extract audio features (MFCC, spectral features) and use classical ML
- **Image-Based Approach**: Convert audio to spectrograms and use CNN models
- **Transfer Learning**: Pre-trained models for enhanced performance
- **Interactive Demo**: Streamlit web app for real-time genre prediction

### **Supported Genres**
- Blues, Classical, Country, Disco, Hip-hop
- Jazz, Metal, Pop, Reggae, Rock

### **Models Implemented**
- **Classical ML**: Random Forest, SVM, Gradient Boosting
- **Deep Learning**: Custom CNN, Transfer Learning (MobileNet, ResNet)
- **Evaluation**: Comprehensive metrics and visualizations

## 📊 Dataset

- **GTZAN Music Genre Dataset**: 1000 audio files
- **10 Genres**: 100 songs per genre
- **Duration**: 30 seconds per track
- **Format**: WAV files

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Music-Genre-Classification
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download dataset**:
```bash
python scripts/download_dataset.py
```

## 🎯 Usage

### 1. Feature-Based Approach
```bash
# Extract audio features
python scripts/feature_extraction.py

# Train classical ML models
python scripts/train_tabular_models.py

# Evaluate performance
python scripts/evaluate_models.py
```

### 2. Image-Based Approach
```bash
# Generate spectrograms
python scripts/create_spectrograms.py

# Train CNN models
python scripts/train_cnn_models.py

# Transfer learning
python scripts/transfer_learning.py
```

### 3. Interactive Demo
```bash
streamlit run app.py
```

## 🔧 Technical Implementation

### **Feature Extraction**
- **MFCCs**: Mel-frequency cepstral coefficients
- **Spectral Features**: Centroid, bandwidth, rolloff
- **Zero Crossing Rate**: Temporal features
- **Chroma Features**: Harmonic content

### **Model Architectures**
- **Random Forest**: Ensemble learning for feature-based approach
- **CNN**: Convolutional layers for spectrogram analysis
- **Transfer Learning**: Pre-trained models for enhanced performance

### **Evaluation Metrics**
- **Accuracy**: Overall classification performance
- **F1-Score**: Balanced precision and recall
- **Confusion Matrix**: Detailed classification analysis
- **ROC Curves**: Model comparison

## 📁 Project Structure

```
Music-Genre-Classification/
├── data/
│   ├── raw/                    # Original GTZAN dataset
│   ├── processed/              # Extracted features
│   └── spectrograms/           # Generated spectrograms
├── models/                     # Trained model files
├── notebooks/                  # Jupyter notebooks
├── scripts/                    # Processing scripts
├── utils/                      # Utility functions
├── app.py                      # Streamlit demo
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🎨 Streamlit App Features

### **Main Interface**
- **File Upload**: Upload audio files for genre prediction
- **Model Selection**: Choose between different approaches
- **Real-time Results**: Instant genre classification
- **Confidence Scores**: Prediction probabilities

### **Visualization**
- **Audio Waveform**: Visual representation of uploaded audio
- **Spectrogram**: Frequency-time representation
- **Model Comparison**: Performance metrics comparison
- **Feature Importance**: Most important audio features

===>(screen shots 1)<===
<img width="1920" height="900" alt="image" src="https://github.com/user-attachments/assets/71FBD487-5251-4226-9769-1B16638E9063" />
<br>

===>(screen shots 2)<===
<img width="1920" height="900" alt="image" src="https://github.com/user-attachments/assets/F003321B-22E5-42FB-AC16-FE69A7B2F6B8" />
<br>

===>(checking output 1)<===
<img width="1920" height="900" alt="image" src="https://github.com/user-attachments/assets/0DEACC4D-1F69-495E-9648-7A0F16DD9C16" />
<br>

===>(checking output 2)<===
<img width="1920" height="900" alt="image" src="https://github.com/user-attachments/assets/13EE3B34-47D2-4F30-86D0-6888D2CB3DF5" />
<br>

===>(checking output 3)<===
<img width="1920" height="900" alt="image" src="https://github.com/user-attachments/assets/90A9A131-2CEC-4151-AD0C-606F5C66A3E5" />
<br>

===>(checking output 4)<===
<img width="1920" height="900" alt="image" src="https://github.com/user-attachments/assets/7903FFC7-9EC6-4FED-8C78-E9246523E340" />
<br>

===>(checking output 5)<===
<img width="1920" height="900" alt="image" src="https://github.com/user-attachments/assets/8CE5A6F5-DA45-48FB-8F48-E1A3015481C7" />
<br>

===>(checking output 6)<===
<img width="1920" height="900" alt="image" src="https://github.com/user-attachments/assets/72304807-B6B6-4A0A-B553-084506C3407E" />
<br>









## 📈 Performance Comparison

| Approach | Model | Accuracy | F1-Score |
|----------|-------|----------|----------|
| Feature-based | Random Forest | TBD | TBD |
| Feature-based | SVM | TBD | TBD |
| Image-based | Custom CNN | TBD | TBD |
| Image-based | Transfer Learning | TBD | TBD |

## 🚀 Getting Started

1. **Run the complete pipeline**:
```bash
# Download and organize dataset
python scripts/download_dataset.py

# Extract features (Feature-based approach)
python scripts/feature_extraction.py

# Create spectrograms (Image-based approach)
python scripts/create_spectrograms.py

# Train all models
python scripts/train_all_models.py

# Launch demo app
streamlit run app.py
```

2. **Interactive Demo**: Open http://localhost:8501 in your browser

## 📚 Learning Outcomes

This project demonstrates:
- **Audio Processing**: Feature extraction from audio signals
- **Computer Vision**: Image-based classification of spectrograms
- **Machine Learning**: Classical ML vs Deep Learning comparison
- **Transfer Learning**: Leveraging pre-trained models
- **Model Evaluation**: Comprehensive performance analysis
- **Web Development**: Interactive Streamlit applications

## 🔮 Future Enhancements

- **Real-time Classification**: Live audio stream processing
- **Multi-label Classification**: Multiple genres per song
- **Advanced Architectures**: Transformer models for audio
- **Data Augmentation**: Audio augmentation techniques
- **Scalability**: Handle larger datasets efficiently

## 📄 License

This project is part of the Elevvo Pathways Machine Learning Internship.

---

**Built with ❤️ using Python, TensorFlow, Librosa, and Streamlit**
