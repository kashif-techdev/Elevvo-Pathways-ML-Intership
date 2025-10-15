# 🚀 Music Genre Classification - Quick Start Guide

## 🎯 **Ready to Run in 3 Steps!**

### **Step 1: Install Python (5 minutes)**

**Option A: Download from python.org (Recommended)**
1. Go to https://www.python.org/downloads/
2. Download Python 3.9 or 3.10
3. **IMPORTANT**: Check "Add Python to PATH" during installation
4. Verify: Open Command Prompt and run `python --version`

**Option B: Use Windows Store**
1. Open Microsoft Store
2. Search for "Python 3.9" or "Python 3.10"
3. Install it
4. Verify: Open Command Prompt and run `python --version`

**Option C: Use winget (if available)**
```bash
winget install Python.Python.3.11
```

### **Step 2: Run the Project (2-3 hours)**

**Option A: Automatic Setup (Recommended)**
```bash
# Navigate to project folder
cd Music-Genre-Classification

# Run the complete setup
setup_project.bat
```

**Option B: Manual Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset
python scripts/download_dataset.py

# Run complete training pipeline
python scripts/train_all_models.py

# Launch demo app
streamlit run app.py
```

### **Step 3: Use the App!**

1. **Upload Audio**: Drag and drop MP3/WAV files
2. **Get Predictions**: See genre classification results
3. **Compare Models**: See which AI models work best
4. **Supported Genres**: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock

## 🎵 **What You'll Get**

- **Interactive Web App**: Real-time genre classification
- **Multiple AI Models**: Feature-based and image-based approaches
- **10 Music Genres**: Complete genre classification system
- **Performance Metrics**: See which models work best
- **Visualizations**: Spectrograms, confidence scores, comparisons

## 📊 **Expected Timeline**

- **Python Installation**: 5-10 minutes
- **Dataset Download**: 10-15 minutes (1.2GB)
- **Feature Extraction**: 30-45 minutes
- **Model Training**: 1-2 hours
- **Total Setup Time**: 2-3 hours

## 🔧 **System Requirements**

- **RAM**: 8GB+ recommended
- **Storage**: 3GB+ free space
- **Python**: 3.8+ required
- **GPU**: Optional but recommended for faster training

## 🎉 **Ready to Start?**

1. **Install Python** (if not already installed)
2. **Run**: `setup_project.bat` (double-click the file)
3. **Wait**: Let it download dataset and train models
4. **Enjoy**: Use the interactive web app!

## 🆘 **Need Help?**

**Common Issues:**
- **Python not found**: Make sure Python is installed and added to PATH
- **Permission errors**: Run Command Prompt as Administrator
- **Memory issues**: Close other applications to free up RAM
- **Slow training**: This is normal - training takes 1-2 hours

**Troubleshooting:**
```bash
# Check Python
python --version

# Check dependencies
pip list

# Test individual components
python scripts/verify_dataset.py
```

## 🎯 **Success Indicators**

You'll know it's working when:
- ✅ Python installs successfully
- ✅ Dependencies install without errors
- ✅ Dataset downloads (1.2GB)
- ✅ Models train successfully
- ✅ Streamlit app opens in browser
- ✅ You can upload audio files and get predictions

**Let's get started! 🚀**
