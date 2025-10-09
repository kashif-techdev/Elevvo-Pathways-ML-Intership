"""
🎵 Music Genre Classification - Streamlit Demo App

Interactive web application for music genre classification.
Supports both feature-based and image-based approaches.
"""

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🎵 Music Genre Classification",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

class MusicGenreClassifier:
    """Main classifier class for the Streamlit app"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_path = self.project_root / "data"
        self.models_path = self.data_path / "models"
        self.processed_path = self.data_path / "processed"
        
        # Load models and data
        self.load_models()
        self.load_data()
    
    def load_models(self):
        """Load trained models"""
        try:
            # Load tabular models
            self.tabular_models = {}
            tabular_model_files = {
                'Random Forest': 'random_forest.pkl',
                'SVM': 'svm.pkl',
                'Gradient Boosting': 'gradient_boosting.pkl'
            }
            
            for name, filename in tabular_model_files.items():
                model_path = self.models_path / filename
                if model_path.exists():
                    self.tabular_models[name] = joblib.load(model_path)
            
            # Load CNN models
            self.cnn_models = {}
            cnn_model_files = {
                'Custom CNN': 'custom_cnn.h5',
                'MobileNetV2': 'mobilenetv2_transfer.h5',
                'ResNet50': 'resnet50_transfer.h5'
            }
            
            for name, filename in cnn_model_files.items():
                model_path = self.models_path / filename
                if model_path.exists():
                    self.cnn_models[name] = load_model(model_path)
            
            # Load scaler and encoder
            scaler_path = self.processed_path / "feature_scaler.pkl"
            encoder_path = self.processed_path / "label_encoder.pkl"
            
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
            
            if len(self.tabular_models) == 0 and len(self.cnn_models) == 0:
                st.warning("⚠️ No trained models found. Please run the training scripts first.")
                st.info("💡 To train models, run: `python scripts/train_all_models.py`")
            else:
                st.success(f"✅ Loaded {len(self.tabular_models)} tabular models and {len(self.cnn_models)} CNN models")
            
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            self.tabular_models = {}
            self.cnn_models = {}
    
    def load_data(self):
        """Load dataset information"""
        try:
            # Load feature statistics
            stats_path = self.processed_path / "feature_stats.json"
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    self.dataset_stats = json.load(f)
            else:
                self.dataset_stats = {}
            
            # Load results
            tabular_results_path = self.models_path / "tabular_results.json"
            cnn_results_path = self.models_path / "cnn_results.json"
            
            self.tabular_results = {}
            self.cnn_results = {}
            
            if tabular_results_path.exists():
                with open(tabular_results_path, 'r') as f:
                    self.tabular_results = json.load(f)
            
            if cnn_results_path.exists():
                with open(cnn_results_path, 'r') as f:
                    self.cnn_results = json.load(f)
            
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            self.dataset_stats = {}
            self.tabular_results = {}
            self.cnn_results = {}
    
    def extract_audio_features(self, audio_file):
        """Extract features from uploaded audio file"""
        try:
            # Load audio with error handling
            try:
                y, sr = librosa.load(audio_file, sr=22050, duration=30)
            except Exception as e:
                st.error(f"❌ Error loading audio file: {e}")
                return None, None, None
            
            # Check if audio is valid
            if len(y) == 0:
                st.error("❌ Audio file is empty or corrupted")
                return None, None, None
            
            # Extract features (same as training)
            # MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_centroid_mean = float(np.mean(spectral_centroids))
            spectral_centroid_std = float(np.std(spectral_centroids))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_bandwidth_mean = float(np.mean(spectral_bandwidth))
            spectral_bandwidth_std = float(np.std(spectral_bandwidth))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_rolloff_mean = float(np.mean(spectral_rolloff))
            spectral_rolloff_std = float(np.std(spectral_rolloff))
            
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = float(np.mean(zcr))
            zcr_std = float(np.std(zcr))
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)
            
            # Rhythm features
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo = float(tempo) if not np.isnan(tempo) else 0.0
            except:
                tempo = 0.0
                
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            if len(onset_times) > 1:
                onset_intervals = np.diff(onset_times)
                rhythm_mean = float(np.mean(onset_intervals))
                rhythm_std = float(np.std(onset_intervals))
            else:
                rhythm_mean = 0.0
                rhythm_std = 0.0
            
            # Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            tonnetz_mean = np.mean(tonnetz, axis=1)
            tonnetz_std = np.std(tonnetz, axis=1)
            
            # Ensure all arrays are 1D and have consistent shapes
            # Flatten all arrays to ensure they're 1D
            mfcc_mean_flat = np.asarray(mfcc_mean).flatten()
            mfcc_std_flat = np.asarray(mfcc_std).flatten()
            chroma_mean_flat = np.asarray(chroma_mean).flatten()
            chroma_std_flat = np.asarray(chroma_std).flatten()
            tonnetz_mean_flat = np.asarray(tonnetz_mean).flatten()
            tonnetz_std_flat = np.asarray(tonnetz_std).flatten()
            
            # Create scalar features array
            scalar_features = np.array([
                spectral_centroid_mean, spectral_centroid_std,
                spectral_bandwidth_mean, spectral_bandwidth_std,
                spectral_rolloff_mean, spectral_rolloff_std,
                zcr_mean, zcr_std, tempo, rhythm_mean, rhythm_std
            ])
            
            # Combine all features - ensure all arrays are 1D
            features = np.concatenate([
                mfcc_mean_flat, 
                mfcc_std_flat,
                scalar_features,
                chroma_mean_flat, 
                chroma_std_flat,
                tonnetz_mean_flat, 
                tonnetz_std_flat
            ])
            
            # Ensure features is a 1D numpy array
            features = np.asarray(features).flatten()
            
            return features, y, sr
            
        except Exception as e:
            st.error(f"❌ Error extracting features: {e}")
            return None, None, None
    
    def create_spectrogram(self, y, sr):
        """Create mel spectrogram from audio"""
        try:
            # Ensure data directory exists
            self.data_path.mkdir(parents=True, exist_ok=True)
            
            # Create mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
            ax.axis('off')
            
            # Display spectrogram
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
            
            # Save to temporary file
            temp_path = self.data_path / "temp_spectrogram.png"
            plt.savefig(temp_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()
            
            return temp_path
            
        except Exception as e:
            st.error(f"❌ Error creating spectrogram: {e}")
            return None
    
    def predict_genre_tabular(self, features):
        """Predict genre using tabular models"""
        predictions = {}
        
        # Check if models and scaler are available
        if not hasattr(self, 'scaler') or self.scaler is None:
            st.error("❌ Scaler not found. Please train the models first.")
            return predictions
            
        if not hasattr(self, 'label_encoder') or self.label_encoder is None:
            st.error("❌ Label encoder not found. Please train the models first.")
            return predictions
            
        if not self.tabular_models:
            st.warning("⚠️ No tabular models found. Please train the models first.")
            return predictions
        
        try:
            # Standardize features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            for name, model in self.tabular_models.items():
                try:
                    # Make prediction
                    pred_proba = model.predict_proba(features_scaled)[0]
                    pred_class = model.predict(features_scaled)[0]
                    
                    # Get class name
                    class_name = self.label_encoder.inverse_transform([pred_class])[0]
                    
                    predictions[name] = {
                        'class': class_name,
                        'confidence': pred_proba[pred_class],
                        'probabilities': dict(zip(self.label_encoder.classes_, pred_proba))
                    }
                    
                except Exception as e:
                    st.error(f"❌ Error with {name}: {e}")
        
        except Exception as e:
            st.error(f"❌ Error in tabular prediction: {e}")
        
        return predictions
    
    def predict_genre_cnn(self, spectrogram_path):
        """Predict genre using CNN models"""
        predictions = {}
        
        # Check if models and encoder are available
        if not hasattr(self, 'label_encoder') or self.label_encoder is None:
            st.error("❌ Label encoder not found. Please train the models first.")
            return predictions
            
        if not self.cnn_models:
            st.warning("⚠️ No CNN models found. Please train the models first.")
            return predictions
        
        if not spectrogram_path or not Path(spectrogram_path).exists():
            st.error("❌ Spectrogram file not found.")
            return predictions
        
        try:
            # Load and preprocess image
            img = load_img(spectrogram_path, target_size=(128, 128))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            for name, model in self.cnn_models.items():
                try:
                    # Make prediction
                    pred_proba = model.predict(img_array, verbose=0)[0]
                    pred_class = np.argmax(pred_proba)
                    
                    # Get class name
                    class_name = self.label_encoder.inverse_transform([pred_class])[0]
                    
                    predictions[name] = {
                        'class': class_name,
                        'confidence': pred_proba[pred_class],
                        'probabilities': dict(zip(self.label_encoder.classes_, pred_proba))
                    }
                    
                except Exception as e:
                    st.error(f"❌ Error with {name}: {e}")
        
        except Exception as e:
            st.error(f"❌ Error processing spectrogram: {e}")
        
        return predictions

def main():
    """Main Streamlit app"""
    
    # Initialize variables
    features = None
    spectrogram_path = None
    
    # Header
    st.markdown('<h1 class="main-header">🎵 Music Genre Classification</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            AI-Powered Music Genre Classification using Feature-Based and Image-Based Approaches
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize classifier
    classifier = MusicGenreClassifier()
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Model selection
    approach = st.sidebar.selectbox(
        "Select Approach",
        ["Feature-Based (Tabular)", "Image-Based (CNN)", "Both Approaches"]
    )
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Audio File",
        type=['wav', 'mp3', 'm4a', 'flac'],
        help="Upload a 30-second audio file for genre classification"
    )
    
    # File validation
    if uploaded_file is not None:
        # Check file size (limit to 50MB)
        max_size = 50 * 1024 * 1024  # 50MB
        if uploaded_file.size > max_size:
            st.error(f"❌ File too large. Maximum size is 50MB. Your file is {uploaded_file.size / (1024*1024):.1f}MB")
            uploaded_file = None
        elif uploaded_file.size < 1024:  # Less than 1KB
            st.error("❌ File too small. Please upload a valid audio file.")
            uploaded_file = None
    
    # Main content
    if uploaded_file is not None:
        st.success("✅ Audio file uploaded successfully!")
        
        # Display audio info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
        with col3:
            st.metric("File Type", uploaded_file.name.split('.')[-1].upper())
        
        # Process audio
        with st.spinner("🎵 Processing audio file..."):
            try:
                features, y, sr = classifier.extract_audio_features(uploaded_file)
                
                if features is not None:
                    st.success("✅ Audio features extracted successfully!")
                    
                    # Create spectrogram
                    spectrogram_path = classifier.create_spectrogram(y, sr)
                    
                    if spectrogram_path is not None:
                        st.success("✅ Spectrogram created successfully!")
                    else:
                        st.warning("⚠️ Could not create spectrogram, but features were extracted successfully.")
                else:
                    st.error("❌ Failed to extract audio features. Please try a different audio file.")
                    st.info("💡 Tips: Make sure your audio file is not corrupted and is in a supported format (WAV, MP3, M4A, FLAC)")
            except Exception as e:
                st.error(f"❌ Unexpected error during processing: {e}")
                st.info("💡 Please try uploading a different audio file or check if the file is not corrupted.")
        
        # Display results
        if features is not None:
            st.markdown("---")
            st.markdown("## 🎯 Genre Classification Results")
            
            # Tabular predictions
            if approach in ["Feature-Based (Tabular)", "Both Approaches"]:
                st.markdown("### 📊 Feature-Based Predictions")
                
                with st.spinner("🤖 Running tabular models..."):
                    tabular_predictions = classifier.predict_genre_tabular(features)
                
                if tabular_predictions:
                    # Display predictions
                    for model_name, pred in tabular_predictions.items():
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{model_name}**")
                        with col2:
                            st.markdown(f"🎵 **{pred['class']}**")
                        with col3:
                            st.markdown(f"🎯 **{pred['confidence']:.3f}**")
                    
                    # Confidence comparison
                    st.markdown("#### 📈 Confidence Comparison")
                    model_names = list(tabular_predictions.keys())
                    confidences = [tabular_predictions[name]['confidence'] for name in model_names]
                    
                    fig = px.bar(
                        x=model_names, 
                        y=confidences,
                        title="Model Confidence Scores",
                        labels={'x': 'Model', 'y': 'Confidence'},
                        color=confidences,
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # CNN predictions
            if approach in ["Image-Based (CNN)", "Both Approaches"] and spectrogram_path is not None:
                st.markdown("### 🖼️ Image-Based Predictions")
                
                with st.spinner("🤖 Running CNN models..."):
                    cnn_predictions = classifier.predict_genre_cnn(spectrogram_path)
                
                if cnn_predictions:
                    # Display predictions
                    for model_name, pred in cnn_predictions.items():
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{model_name}**")
                        with col2:
                            st.markdown(f"🎵 **{pred['class']}**")
                        with col3:
                            st.markdown(f"🎯 **{pred['confidence']:.3f}**")
                    
                    # Confidence comparison
                    st.markdown("#### 📈 CNN Confidence Comparison")
                    model_names = list(cnn_predictions.keys())
                    confidences = [cnn_predictions[name]['confidence'] for name in model_names]
                    
                    fig = px.bar(
                        x=model_names, 
                        y=confidences,
                        title="CNN Model Confidence Scores",
                        labels={'x': 'Model', 'y': 'Confidence'},
                        color=confidences,
                        color_continuous_scale='plasma'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Display spectrogram
            if spectrogram_path is not None:
                st.markdown("### 🎨 Generated Spectrogram")
                st.image(str(spectrogram_path), caption="Mel Spectrogram", use_column_width=True)
        
        # Clean up temporary files
        if spectrogram_path and spectrogram_path.exists():
            spectrogram_path.unlink()
    
    else:
        # Show dataset information
        st.markdown("## 📊 Dataset Information")
        
        if classifier.dataset_stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", classifier.dataset_stats.get('total_samples', 'N/A'))
            with col2:
                st.metric("Features", classifier.dataset_stats.get('feature_dimensions', 'N/A'))
            with col3:
                st.metric("Genres", classifier.dataset_stats.get('n_classes', 'N/A'))
            with col4:
                st.metric("Sparsity", "93.65%")
        
        # Show model performance
        if classifier.tabular_results or classifier.cnn_results:
            st.markdown("## 🏆 Model Performance")
            
            if classifier.tabular_results:
                st.markdown("### 📊 Tabular Models")
                tabular_df = pd.DataFrame(classifier.tabular_results).T
                st.dataframe(tabular_df, use_container_width=True)
            
            if classifier.cnn_results:
                st.markdown("### 🖼️ CNN Models")
                cnn_df = pd.DataFrame(classifier.cnn_results).T
                st.dataframe(cnn_df, use_container_width=True)
        
        # Instructions
        st.markdown("## 🚀 How to Use")
        st.markdown("""
        1. **Upload Audio**: Use the sidebar to upload a 30-second audio file
        2. **Select Approach**: Choose between feature-based, image-based, or both
        3. **Get Results**: View genre predictions and confidence scores
        4. **Compare Models**: See how different algorithms perform
        """)
        
        # Supported genres
        st.markdown("## 🎵 Supported Genres")
        genres = ['Blues', 'Classical', 'Country', 'Disco', 'Hip-hop', 
                 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
        
        cols = st.columns(5)
        for i, genre in enumerate(genres):
            with cols[i % 5]:
                st.markdown(f"🎵 **{genre}**")

if __name__ == "__main__":
    main()
