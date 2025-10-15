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

### **screen shots**
<img width="1920" height="997" alt="{730B8462-A731-42D6-B4DC-3CFB514976E1}" src="https://github.com/user-attachments/assets/e7f85257-a14c-402b-bd04-dd15de719d0b" />
<br><br>
<img width="1920" height="995" alt="{DBD27BBA-3689-40E0-A7A1-0CA8BD6790EA}" src="https://github.com/user-attachments/assets/12897cac-d7da-482d-b2bb-2f38a4320658" />
<br><br>
### **cheking output**
<img width="1920" height="999" alt="{780E4C01-4B2D-497C-A40F-6E79A4FC0784}" src="https://github.com/user-attachments/assets/4646858a-fb98-4a8f-b9be-558abec87f32" />
<br><br>
<img width="1920" height="997" alt="{D0EA1094-0290-41EA-8C2B-F4E9990C05EA}" src="https://github.com/user-attachments/assets/e35c0310-887b-4c97-bce9-5a38b8a85333" />
<br><br>
<img width="1920" height="1010" alt="{C78B91D8-4380-48F8-9465-65BC03709437}" src="https://github.com/user-attachments/assets/dccfe513-42f0-48d4-8a29-072cf9ae29d8" />
<br><br>
<img width="1920" height="999" alt="{C6A25E20-8A90-45B9-9BDE-1F4B79595DB8}" src="https://github.com/user-attachments/assets/5bf40789-716c-433a-a178-b9d58e39a687" />
<br><br>
<img width="1920" height="996" alt="{DC40C656-1C9C-4BFF-95A1-E76CF8E8461E}" src="https://github.com/user-attachments/assets/de9781b5-e525-47b5-a7c2-646b3971a74e" />
<br><br>
<img width="1920" height="997" alt="{C8372634-19B7-4DB8-9A20-E7AB7A23B78F}" src="https://github.com/user-attachments/assets/78b11d4d-ca52-441f-959d-e6001e6b46a2" />










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
