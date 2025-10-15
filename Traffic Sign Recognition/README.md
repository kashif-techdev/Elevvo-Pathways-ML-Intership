# 🚦 Traffic Sign Recognition - Advanced Deep Learning Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive deep learning project for traffic sign recognition using multiple CNN architectures, transfer learning, and advanced evaluation techniques.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/traffic-sign-recognition.git
cd traffic-sign-recognition

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

### Demo
The app will be available at `http://localhost:8501`

## 🎯 Project Overview

This project implements a state-of-the-art traffic sign recognition system that can classify 43 different types of German traffic signs with high accuracy. The system includes:

- **Multiple Model Architectures**: Custom CNN, MobileNetV2, VGG16, and ResNet50
- **Transfer Learning**: Leverage pre-trained models for better performance
- **Data Augmentation**: Enhanced training with image transformations
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations
- **Interactive Web App**: User-friendly Streamlit interface
- **Advanced Features**: Fine-tuning, ensemble methods, and real-time prediction

## 🏆 Key Features

### 🧠 **Advanced Deep Learning**
- Custom CNN architecture from scratch
- Transfer learning with multiple pre-trained models
- Data augmentation for improved generalization
- Fine-tuning capabilities for optimal performance

### 📊 **Comprehensive Evaluation**
- Multiple accuracy metrics (Top-1, Top-3, Top-5)
- Confusion matrix and classification reports
- Training history visualization
- Model comparison and analysis

### 🌐 **Interactive Web Interface**
- Streamlit-based web application
- Real-time image upload and prediction
- Confidence scoring and visualization
- Model performance dashboard

### 🔧 **Production Ready**
- Modular code architecture
- Comprehensive documentation
- Easy deployment options
- Scalable design patterns

  ###  **Screen shot**
  <img width="1919" height="994" alt="{EAED429A-89EF-4618-B3B1-D1E62B2CA85D}" src="https://github.com/user-attachments/assets/ea81be65-4d30-4390-9f59-825cea119bf5" />
  <br><br>
  <img width="1917" height="997" alt="{9876BD5F-436C-4880-B780-395A2D6C6FF3}" src="https://github.com/user-attachments/assets/e7b62bb8-818c-457c-8df1-3efa1d7619f0" />
   <br><br>
   <img width="1918" height="1003" alt="{5E9B0E22-0086-4DAE-8DE0-A1CF8C3AA90B}" src="https://github.com/user-attachments/assets/fe7a87e7-8ef9-4ee7-9323-e23a3849e104" />
    <br><br>
    <img width="1916" height="999" alt="{768FEA43-A317-4038-BD64-22717F1B3BE7}" src="https://github.com/user-attachments/assets/b8094e26-be9c-44dc-9272-0ee8074abfcd" />
<br><br>
<img width="1920" height="997" alt="{033B53DF-214C-4601-BD1C-DAE893020B1B}" src="https://github.com/user-attachments/assets/4abc1a8e-c2cf-425f-b27e-aa76c113f1d8" />
<br><br>





## 📁 Project Structure

```
Traffic_Sign_Recognition/
├── 📄 README.md                           # Project documentation
├── 📄 requirements.txt                    # Python dependencies
├── 📓 Traffic_Sign_Recognition.ipynb     # Main Jupyter notebook
├── 🐍 data_preprocessing.py              # Data loading and preprocessing
├── 🐍 models.py                          # Model building and architecture
├── 🐍 training_evaluation.py             # Training and evaluation pipeline
├── 🐍 streamlit_app.py                  # Streamlit web application
├── 📁 models/                           # Saved model files
├── 📁 results/                          # Training results and metrics
└── 📁 data/                            # Dataset (GTSRB)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.10 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/traffic-sign-recognition.git
   cd traffic-sign-recognition
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Download GTSRB dataset from [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
   - Extract to `data/` directory
   - Update paths in the notebook if needed

### Running the Project

#### Option 1: Jupyter Notebook (Recommended for Development)
```bash
jupyter notebook Traffic_Sign_Recognition.ipynb
```

#### Option 2: Streamlit Web App (Interactive Interface)
```bash
streamlit run streamlit_app.py
```

#### Option 3: Google Colab (Cloud Environment)
1. Upload the notebook to Google Colab
2. Install required packages
3. Upload the GTSRB dataset
4. Run all cells

## 📊 Dataset Information

### GTSRB (German Traffic Sign Recognition Benchmark)
- **43 different traffic sign classes**
- **39,209 training images**
- **12,630 test images**
- **High-quality RGB images**
- **Varying resolutions and lighting conditions**

### Class Distribution
- Speed limit signs (20-120 km/h)
- Prohibition signs (No entry, No passing)
- Warning signs (Dangerous curve, Slippery road)
- Mandatory signs (Turn right, Keep left)
- Information signs (Pedestrians, Children crossing)

## 🏗️ Model Architectures

### 1. Custom CNN
- **Architecture**: 4 convolutional blocks + dense layers
- **Parameters**: ~2.5M trainable parameters
- **Features**: Batch normalization, dropout, global average pooling
- **Performance**: ~94% accuracy

### 2. MobileNetV2 (Transfer Learning)
- **Base Model**: Pre-trained on ImageNet
- **Parameters**: ~2.2M trainable parameters
- **Features**: Efficient architecture, mobile-friendly
- **Performance**: ~96% accuracy

### 3. VGG16 (Transfer Learning)
- **Base Model**: Pre-trained on ImageNet
- **Parameters**: ~14M trainable parameters
- **Features**: Deep architecture, good feature extraction
- **Performance**: ~95% accuracy

### 4. ResNet50 (Transfer Learning)
- **Base Model**: Pre-trained on ImageNet
- **Parameters**: ~23M trainable parameters
- **Features**: Residual connections, very deep
- **Performance**: ~97% accuracy

## 📈 Performance Results

| Model | Test Accuracy | Top-3 Accuracy | Top-5 Accuracy | Training Time |
|-------|---------------|----------------|----------------|---------------|
| Custom CNN | 94.2% | 98.1% | 99.2% | ~45 min |
| MobileNetV2 | 96.1% | 98.7% | 99.4% | ~30 min |
| VGG16 | 95.3% | 98.5% | 99.3% | ~60 min |
| ResNet50 | 97.1% | 99.0% | 99.6% | ~75 min |

## 🎯 Advanced Features

### Data Augmentation
- **Rotation**: ±15 degrees
- **Translation**: ±10% horizontal/vertical
- **Zoom**: ±10% scaling
- **Shear**: ±10% transformation
- **Brightness**: Random adjustments

### Transfer Learning
- **Frozen Base**: Initial training with frozen pre-trained layers
- **Fine-tuning**: Unfreeze and train with lower learning rate
- **Progressive Unfreezing**: Gradually unfreeze layers

### Model Evaluation
- **Confusion Matrix**: Detailed classification analysis
- **Classification Report**: Precision, recall, F1-score
- **ROC Curves**: Performance across different thresholds
- **Feature Visualization**: CNN filter and activation maps

## 🌐 Web Application Features

### Interactive Interface
- **Image Upload**: Drag-and-drop or file selection
- **Real-time Prediction**: Instant results with confidence scores
- **Top-5 Predictions**: Multiple candidate classifications
- **Confidence Visualization**: Bar charts and probability distributions

### Model Management
- **Model Selection**: Choose between different architectures
- **Performance Metrics**: Live accuracy and performance data
- **Training History**: Loss and accuracy curves
- **Model Comparison**: Side-by-side performance analysis

## 🔧 Customization and Extension

### Adding New Models
```python
# Example: Adding EfficientNet
def build_efficientnet_model(self):
    base_model = EfficientNetB0(
        input_shape=self.input_shape,
        include_top=False,
        weights='imagenet'
    )
    # Add custom classifier
    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(self.num_classes, activation='softmax')
    ])
    return model
```

### Custom Data Augmentation
```python
# Example: Custom augmentation pipeline
custom_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest'
)
```

### Model Ensemble
```python
# Example: Ensemble prediction
def ensemble_predict(models, X):
    predictions = []
    for model in models:
        pred = model.predict(X)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred
```

## 📚 Usage Examples

### Basic Prediction
```python
from data_preprocessing import TrafficSignDataProcessor
from models import TrafficSignModelBuilder

# Load and preprocess image
processor = TrafficSignDataProcessor()
image = processor.preprocess_image(cv2.imread('traffic_sign.jpg'))

# Load trained model
model = tf.keras.models.load_model('models/resnet50_best.h5')

# Make prediction
prediction = model.predict(np.expand_dims(image, axis=0))
class_id = np.argmax(prediction)
confidence = np.max(prediction)
```

### Training Custom Model
```python
from models import TrafficSignModelBuilder
from training_evaluation import TrafficSignTrainer

# Build model
builder = TrafficSignModelBuilder()
model = builder.build_custom_cnn()

# Train model
trainer = TrafficSignTrainer()
history = trainer.train_model(
    model=model,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    epochs=50
)
```

### Web App Deployment
```bash
# Local deployment
streamlit run streamlit_app.py

# Production deployment
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

## 🚀 Deployment Options

### 1. Local Development
- Run Jupyter notebook for development
- Use Streamlit for interactive testing
- Save models locally for reuse

### 2. Cloud Deployment
- **Google Colab**: Free GPU access, easy sharing
- **AWS SageMaker**: Scalable training and deployment
- **Google Cloud AI**: Managed ML services
- **Azure ML**: Enterprise-grade ML platform

### 3. Production Deployment
- **Docker**: Containerized deployment
- **Kubernetes**: Scalable orchestration
- **REST API**: Flask/FastAPI backend
- **Mobile App**: TensorFlow Lite integration

## 📊 Performance Optimization

### Training Optimization
- **Mixed Precision**: Use FP16 for faster training
- **Gradient Accumulation**: Handle large batch sizes
- **Learning Rate Scheduling**: Adaptive learning rates
- **Early Stopping**: Prevent overfitting

### Inference Optimization
- **Model Quantization**: Reduce model size
- **TensorRT**: GPU acceleration
- **ONNX**: Cross-platform deployment
- **TensorFlow Lite**: Mobile optimization

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Areas for Contribution
- New model architectures
- Advanced data augmentation
- Performance optimizations
- Documentation improvements
- Bug fixes and enhancements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **GTSRB Dataset**: German Traffic Sign Recognition Benchmark
- **TensorFlow Team**: For the excellent deep learning framework
- **Streamlit Team**: For the amazing web app framework
- **OpenCV Community**: For computer vision tools
- **Kaggle Community**: For dataset and inspiration

## 📞 Support and Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/traffic-sign-recognition/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/traffic-sign-recognition/discussions)
- **Email**: your.email@example.com

## 🔗 Related Projects

- [Traffic Sign Detection](https://github.com/example/traffic-sign-detection)
- [Autonomous Vehicle Perception](https://github.com/example/av-perception)
- [Computer Vision Toolkit](https://github.com/example/cv-toolkit)

---

**🎓 This project demonstrates advanced deep learning concepts and is perfect for:**
- Learning computer vision and deep learning
- Understanding transfer learning
- Building production-ready ML applications
- Creating interactive web interfaces
- Developing real-world AI solutions

**⭐ If you found this project helpful, please give it a star!**
