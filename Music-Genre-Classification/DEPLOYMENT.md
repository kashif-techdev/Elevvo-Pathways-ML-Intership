# 🚀 Music Genre Classification - Deployment Guide

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning)

## 🛠️ Local Deployment

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/music-genre-classification.git
cd music-genre-classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application
```bash
# For enhanced app with CNN features
streamlit run app_enhanced.py

# For basic app
streamlit run app.py
```

### 4. Access Application
- Local URL: `http://localhost:8501`
- Network URL: `http://your-ip:8501`

## ☁️ Cloud Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy automatically

### Heroku
1. Add `Procfile`:
```
web: streamlit run app_enhanced.py --server.port=$PORT --server.address=0.0.0.0
```
2. Deploy using Heroku CLI

### Docker
1. Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app_enhanced.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
2. Build and run:
```bash
docker build -t music-genre-classification .
docker run -p 8501:8501 music-genre-classification
```

## 🔧 Configuration

### Environment Variables
- `TF_CPP_MIN_LOG_LEVEL=2` (suppress TensorFlow warnings)
- `TF_ENABLE_ONEDNN_OPTS=0` (disable oneDNN optimizations)

### Model Files
- Place trained models in `data/models/` directory
- Supported formats: `.pkl` (scikit-learn models)

### Data Requirements
- Audio files should be in WAV format
- Supported genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- Audio length: 30 seconds (recommended)

## 📊 Performance Optimization

### For Production
1. Use GPU-enabled TensorFlow
2. Implement model caching
3. Add request rate limiting
4. Use CDN for static assets

### Monitoring
1. Add logging
2. Implement health checks
3. Monitor memory usage
4. Track prediction accuracy

## 🚨 Troubleshooting

### Common Issues
1. **Model not found**: Ensure model files are in `data/models/` directory
2. **Audio format issues**: Convert to WAV format
3. **Memory issues**: Reduce batch size or use smaller models
4. **Import errors**: Check all dependencies are installed

### Support
- Check logs for detailed error messages
- Verify all requirements are met
- Test with sample audio files first
