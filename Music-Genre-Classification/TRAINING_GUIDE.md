# 🎵 Music Genre Classification - Training Guide

## 📋 Current Status

✅ **Completed Steps:**
1. ✅ Dataset downloaded and organized (1000 audio files)
2. ✅ Feature extraction completed (999/1000 files, 87 features)
3. ✅ Train/test split created (799 train, 200 test samples)
4. ✅ Spectrograms generated (999/1000 files, 99.9% success)
5. ✅ Partial tabular model training (Random Forest: 70.5%, SVM: 72.5%)

❌ **Interrupted:** Gradient Boosting training (takes too long without GPU)

## 🚀 Next Steps for GPU Training

### 1. **Complete Tabular Models Training**
```bash
# Run on your GPU machine
python scripts/train_tabular_models.py
```
**Expected time:** 10-15 minutes with GPU
**Models to train:** Random Forest, SVM, Gradient Boosting, XGBoost, LightGBM, CatBoost

### 2. **Train CNN Models**
```bash
python scripts/train_cnn_models.py
```
**Expected time:** 30-45 minutes with GPU
**Models to train:** Custom CNN, MobileNetV2, ResNet50

### 3. **Transfer Learning Models**
```bash
python scripts/transfer_learning.py
```
**Expected time:** 20-30 minutes with GPU
**Models to train:** Pre-trained models with fine-tuning

### 4. **Model Evaluation**
```bash
python scripts/evaluate_models.py
```
**Expected time:** 5-10 minutes
**Output:** Performance metrics and comparisons

### 5. **Compare Approaches**
```bash
python scripts/compare_approaches.py
```
**Expected time:** 2-5 minutes
**Output:** Comprehensive comparison report

## 📁 Files to Transfer

Copy these files to your GPU machine:
```
Music-Genre-Classification/
├── data/
│   ├── raw/                    # 1000 audio files (already organized)
│   └── processed/              # Features and splits
│       ├── audio_features.pkl
│       ├── audio_features.csv
│       ├── train_test_split.pkl
│       ├── feature_scaler.pkl
│       ├── label_encoder.pkl
│       └── feature_stats.json
├── scripts/                    # All training scripts
├── app.py                      # Streamlit app
└── requirements.txt
```

## 🔧 GPU Setup Requirements

### Install Dependencies:
```bash
pip install -r requirements.txt
pip install tensorflow-gpu  # For GPU support
pip install torch torchvision torchaudio  # For PyTorch models
```

### Verify GPU Setup:
```python
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
```

## 📊 Expected Results

### Tabular Models Performance:
- **Random Forest:** ~70-75% accuracy
- **SVM:** ~72-77% accuracy  
- **Gradient Boosting:** ~75-80% accuracy
- **XGBoost:** ~78-82% accuracy
- **LightGBM:** ~77-81% accuracy
- **CatBoost:** ~76-80% accuracy

### CNN Models Performance:
- **Custom CNN:** ~65-70% accuracy
- **MobileNetV2:** ~70-75% accuracy
- **ResNet50:** ~72-77% accuracy

## 🎯 Final Steps

### 1. **Run Complete Training Pipeline**
```bash
python scripts/train_all_models.py
```

### 2. **Launch the App**
```bash
streamlit run app.py
```

### 3. **Test with Your Audio Files**
- Upload MP3/WAV files
- Get genre predictions
- Compare different model approaches

## 📈 Performance Optimization

### For Faster Training:
1. **Reduce dataset size** (if needed):
   ```python
   # In feature_extraction.py, limit files per genre
   audio_files = audio_files[:50]  # Use only 50 files per genre
   ```

2. **Use smaller models**:
   ```python
   # In train_cnn_models.py, reduce model complexity
   model.add(Conv2D(32, (3, 3), activation='relu'))  # Instead of 64
   ```

3. **Enable mixed precision** (for TensorFlow):
   ```python
   tf.keras.mixed_precision.set_global_policy('mixed_float16')
   ```

## 🐛 Troubleshooting

### Common Issues:
1. **CUDA out of memory:** Reduce batch size or model complexity
2. **Slow training:** Check if GPU is being used
3. **Model not loading:** Ensure all dependencies are installed

### Debug Commands:
```bash
# Check GPU usage
nvidia-smi

# Test individual components
python scripts/verify_dataset.py
python scripts/feature_extraction.py  # Test feature extraction
```

## 📝 Notes

- **Dataset:** GTZAN Music Genre Dataset (1000 files, 10 genres)
- **Features:** 87 audio features (MFCC, spectral, chroma, rhythm, tonnetz)
- **Train/Test Split:** 80/20 stratified split
- **Cross-validation:** 5-fold CV for robust evaluation
- **Genres:** blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

## 🎉 Success Criteria

Training is complete when you have:
- ✅ All tabular models trained and saved
- ✅ All CNN models trained and saved  
- ✅ Transfer learning models completed
- ✅ Model evaluation results generated
- ✅ Streamlit app working with predictions

## 📞 Support

If you encounter issues:
1. Check the error messages carefully
2. Verify GPU setup and dependencies
3. Ensure all data files are present
4. Try running individual scripts first

**Good luck with your GPU training! 🚀**
