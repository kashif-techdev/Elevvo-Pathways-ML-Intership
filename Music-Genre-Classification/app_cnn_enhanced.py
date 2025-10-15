"""
🎵 Music Genre Classification - Enhanced CNN Streamlit App

Advanced Streamlit application with both feature-based and CNN-based approaches.
Includes real-time spectrogram generation and CNN model predictions.
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
import time
import cv2
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🎵 Music Genre Classification - CNN Enhanced",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(240, 147, 251, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(240, 147, 251, 0.4);
    }
    
    .cnn-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .cnn-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(79, 172, 254, 0.4);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

class EnhancedMusicGenreClassifier:
    """Enhanced classifier with both tabular and CNN approaches"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_path = self.project_root / "data"
        self.models_path = self.data_path / "models"
        self.processed_path = self.data_path / "processed"
        
        # Load models and data
        self.load_models()
        self.load_data()
    
    def load_models(self):
        """Load all trained models with enhanced feedback"""
        try:
            # Load tabular models
            self.tabular_models = {}
            tabular_model_files = {
                'Random Forest': 'random_forest.pkl',
                'SVM': 'svm.pkl',
                'Gradient Boosting': 'gradient_boosting.pkl',
                'Logistic Regression': 'logistic_regression.pkl',
                'K-Nearest Neighbors': 'k-nearest_neighbors.pkl',
                'Naive Bayes': 'naive_bayes.pkl'
            }
            
            loaded_tabular = 0
            for name, filename in tabular_model_files.items():
                model_path = self.models_path / filename
                if model_path.exists():
                    self.tabular_models[name] = joblib.load(model_path)
                    loaded_tabular += 1
            
            # Load CNN models
            self.cnn_models = {}
            cnn_model_files = {
                'Custom CNN': 'custom_cnn.h5',
                'MobileNetV2': 'mobilenetv2_transfer.h5',
                'ResNet50': 'resnet50_transfer.h5',
                'EfficientNetB0': 'efficientnetb0_transfer.h5',
                'DenseNet121': 'densenet121_transfer.h5'
            }
            
            loaded_cnn = 0
            for name, filename in cnn_model_files.items():
                model_path = self.models_path / filename
                if model_path.exists():
                    try:
                        self.cnn_models[name] = load_model(model_path)
                        loaded_cnn += 1
                    except Exception as e:
                        st.warning(f"⚠️ Could not load {name}: {e}")
            
            # Load scaler and encoder
            scaler_path = self.processed_path / "feature_scaler.pkl"
            encoder_path = self.processed_path / "label_encoder.pkl"
            
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
            
            # Display loading results
            if loaded_tabular == 0 and loaded_cnn == 0:
                st.error("🚨 **No trained models found!**")
                st.info("💡 **To train models, run:** `python scripts/train_all_models.py`")
            else:
                st.success(f"✅ **Loaded {loaded_tabular} tabular models and {loaded_cnn} CNN models**")
            
        except Exception as e:
            st.error(f"❌ **Error loading models:** {e}")
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
            
        except Exception as e:
            st.error(f"❌ **Error loading data:** {e}")
            self.dataset_stats = {}
    
    def extract_audio_features(self, audio_file):
        """Extract features from uploaded audio file with progress feedback"""
        try:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🎵 Loading audio file...")
            progress_bar.progress(10)
            
            # Load audio with error handling
            try:
                y, sr = librosa.load(audio_file, sr=22050, duration=30)
            except Exception as e:
                st.error(f"❌ **Error loading audio file:** {e}")
                return None, None, None
            
            # Check if audio is valid
            if len(y) == 0:
                st.error("❌ **Audio file is empty or corrupted**")
                return None, None, None
            
            status_text.text("🔍 Extracting MFCC features...")
            progress_bar.progress(30)
            
            # Extract features (same as training)
            # MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            status_text.text("📊 Computing spectral features...")
            progress_bar.progress(50)
            
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
            
            status_text.text("🎼 Analyzing chroma and rhythm...")
            progress_bar.progress(70)
            
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
            
            status_text.text("🎹 Computing tonnetz features...")
            progress_bar.progress(90)
            
            # Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            tonnetz_mean = np.mean(tonnetz, axis=1)
            tonnetz_std = np.std(tonnetz, axis=1)
            
            # Ensure all arrays are 1D and have consistent shapes
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
            
            # Combine all features
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
            
            status_text.text("✅ Features extracted successfully!")
            progress_bar.progress(100)
            
            # Clear progress indicators
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            return features, y, sr
            
        except Exception as e:
            st.error(f"❌ **Error extracting features:** {e}")
            return None, None, None
    
    def create_enhanced_spectrogram(self, y, sr):
        """Create enhanced spectrogram from audio"""
        try:
            # Ensure data directory exists
            self.data_path.mkdir(parents=True, exist_ok=True)
            
            # Create mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            # Create figure with enhanced styling
            fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
            ax.axis('off')
            
            # Display spectrogram with better colormap
            img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', 
                                        ax=ax, cmap='viridis')
            
            # Add colorbar
            plt.colorbar(img, ax=ax, format='%+2.0f dB')
            
            # Save to temporary file
            temp_path = self.data_path / "temp_spectrogram.png"
            plt.savefig(temp_path, bbox_inches='tight', pad_inches=0.1, dpi=120)
            plt.close()
            
            return temp_path
            
        except Exception as e:
            st.error(f"❌ **Error creating spectrogram:** {e}")
            return None
    
    def predict_genre_tabular(self, features):
        """Predict genre using tabular models"""
        predictions = {}
        
        # Check if models and scaler are available
        if not hasattr(self, 'scaler') or self.scaler is None:
            st.error("❌ **Scaler not found. Please train the models first.**")
            return predictions
            
        if not hasattr(self, 'label_encoder') or self.label_encoder is None:
            st.error("❌ **Label encoder not found. Please train the models first.**")
            return predictions
            
        if not self.tabular_models:
            st.warning("⚠️ **No tabular models found. Please train the models first.**")
            return predictions
        
        try:
            # Standardize features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Create progress tracking
            total_models = len(self.tabular_models)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (name, model) in enumerate(self.tabular_models.items()):
                status_text.text(f"🤖 Running {name}...")
                progress_bar.progress((i + 1) / total_models)
                
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
                    st.error(f"❌ **Error with {name}:** {e}")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
        
        except Exception as e:
            st.error(f"❌ **Error in tabular prediction:** {e}")
        
        return predictions
    
    def predict_genre_cnn(self, spectrogram_path):
        """Predict genre using CNN models"""
        predictions = {}
        
        # Check if models and encoder are available
        if not hasattr(self, 'label_encoder') or self.label_encoder is None:
            st.error("❌ **Label encoder not found. Please train the models first.**")
            return predictions
            
        if not self.cnn_models:
            st.warning("⚠️ **No CNN models found. Please train the models first.**")
            return predictions
        
        if not spectrogram_path or not Path(spectrogram_path).exists():
            st.error("❌ **Spectrogram file not found.**")
            return predictions
        
        try:
            # Load and preprocess image
            img = load_img(spectrogram_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Create progress tracking
            total_models = len(self.cnn_models)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (name, model) in enumerate(self.cnn_models.items()):
                status_text.text(f"🧠 Running {name}...")
                progress_bar.progress((i + 1) / total_models)
                
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
                    st.error(f"❌ **Error with {name}:** {e}")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
        
        except Exception as e:
            st.error(f"❌ **Error processing spectrogram:** {e}")
        
        return predictions

def create_enhanced_visualizations(tabular_predictions, cnn_predictions):
    """Create enhanced visualizations for both approaches"""
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Comparison", "🎯 Confidence Scores", "📈 Probability Distribution", "🏆 Ensemble Analysis"])
    
    with tab1:
        # Model comparison
        if tabular_predictions and cnn_predictions:
            # Combine all predictions
            all_predictions = {**tabular_predictions, **cnn_predictions}
            
            model_names = list(all_predictions.keys())
            confidences = [all_predictions[name]['confidence'] for name in model_names]
            
            # Color by approach
            colors = []
            for name in model_names:
                if name in tabular_predictions:
                    colors.append('#667eea')  # Blue for tabular
                else:
                    colors.append('#4facfe')  # Light blue for CNN
            
            fig = px.bar(
                x=model_names, 
                y=confidences,
                title="🎯 All Models Confidence Comparison",
                labels={'x': 'Model', 'y': 'Confidence'},
                color=model_names,
                color_discrete_sequence=colors,
                height=500
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                title_font_size=18
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Confidence scores comparison
        if tabular_predictions:
            st.markdown("#### 📊 Tabular Models Confidence")
            tabular_names = list(tabular_predictions.keys())
            tabular_confidences = [tabular_predictions[name]['confidence'] for name in tabular_names]
            
            fig = px.bar(
                x=tabular_names, 
                y=tabular_confidences,
                title="Tabular Models Confidence",
                color=tabular_confidences,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if cnn_predictions:
            st.markdown("#### 🧠 CNN Models Confidence")
            cnn_names = list(cnn_predictions.keys())
            cnn_confidences = [cnn_predictions[name]['confidence'] for name in cnn_names]
            
            fig = px.bar(
                x=cnn_names, 
                y=cnn_confidences,
                title="CNN Models Confidence",
                color=cnn_confidences,
                color_continuous_scale='plasma'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Probability distribution
        if tabular_predictions:
            best_tabular = max(tabular_predictions.keys(), key=lambda x: tabular_predictions[x]['confidence'])
            best_pred = tabular_predictions[best_tabular]
            
            genres = list(best_pred['probabilities'].keys())
            probs = list(best_pred['probabilities'].values())
            
            fig = px.pie(
                values=probs,
                names=genres,
                title=f"🎵 Tabular Model Probability Distribution ({best_tabular})",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if cnn_predictions:
            best_cnn = max(cnn_predictions.keys(), key=lambda x: cnn_predictions[x]['confidence'])
            best_pred = cnn_predictions[best_cnn]
            
            genres = list(best_pred['probabilities'].keys())
            probs = list(best_pred['probabilities'].values())
            
            fig = px.pie(
                values=probs,
                names=genres,
                title=f"🧠 CNN Model Probability Distribution ({best_cnn})",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Ensemble analysis
        if tabular_predictions and cnn_predictions:
            st.markdown("#### 🏆 Ensemble Analysis")
            
            # Calculate ensemble predictions
            all_predictions = {**tabular_predictions, **cnn_predictions}
            
            # Get all probability distributions
            ensemble_probs = {}
            for model_name, pred in all_predictions.items():
                for genre, prob in pred['probabilities'].items():
                    if genre not in ensemble_probs:
                        ensemble_probs[genre] = []
                    ensemble_probs[genre].append(prob)
            
            # Calculate average probabilities
            ensemble_avg = {}
            for genre, probs in ensemble_probs.items():
                ensemble_avg[genre] = np.mean(probs)
            
            # Find ensemble prediction
            ensemble_genre = max(ensemble_avg.keys(), key=lambda x: ensemble_avg[x])
            ensemble_confidence = ensemble_avg[ensemble_genre]
            
            # Display ensemble results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🏆 Ensemble Prediction</h4>
                    <h3>🎵 {ensemble_genre}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📊 Ensemble Confidence</h4>
                    <h3>{ensemble_confidence:.3f}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🤖 Models Used</h4>
                    <h3>{len(all_predictions)}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Ensemble probability distribution
            genres = list(ensemble_avg.keys())
            probs = list(ensemble_avg.values())
            
            fig = px.bar(
                x=genres,
                y=probs,
                title="🏆 Ensemble Probability Distribution",
                color=probs,
                color_continuous_scale='viridis'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Enhanced main Streamlit app with CNN integration"""
    
    # Enhanced Header
    st.markdown('<h1 class="main-header">🎵 Music Genre Classification - CNN Enhanced</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem; color: #6c757d; font-size: 1.3rem;">
        🚀 AI-Powered Music Genre Classification using Feature-Based and CNN-Based Approaches
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize classifier
    classifier = EnhancedMusicGenreClassifier()
    
    # Enhanced Sidebar
    st.sidebar.markdown("## 🎛️ **Control Panel**")
    st.sidebar.markdown("---")
    
    # Model selection
    approach = st.sidebar.selectbox(
        "🎯 **Select Classification Approach**",
        ["Feature-Based (Tabular)", "CNN-Based (Deep Learning)", "Both Approaches", "Ensemble Analysis"],
        help="Choose between feature-based machine learning, CNN-based deep learning, or both"
    )
    
    st.sidebar.markdown("---")
    
    # File upload
    st.sidebar.markdown("### 📁 **Upload Audio File**")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a', 'flac'],
        help="Upload a 30-second audio file for genre classification",
        label_visibility="collapsed"
    )
    
    # File validation
    if uploaded_file is not None:
        # Check file size (limit to 50MB)
        max_size = 50 * 1024 * 1024  # 50MB
        if uploaded_file.size > max_size:
            st.error(f"❌ **File too large!** Maximum size is 50MB. Your file is {uploaded_file.size / (1024*1024):.1f}MB")
            uploaded_file = None
        elif uploaded_file.size < 1024:  # Less than 1KB
            st.error("❌ **File too small!** Please upload a valid audio file.")
            uploaded_file = None
    
    # Main content
    if uploaded_file is not None:
        # Success message with animation
        st.success("✅ **Audio file uploaded successfully!**")
        
        # Enhanced file info display
        st.markdown("### 📋 **File Information**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📄 File Name</h4>
                <p>{uploaded_file.name}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>💾 File Size</h4>
                <p>{uploaded_file.size / 1024:.1f} KB</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎵 File Type</h4>
                <p>{uploaded_file.name.split('.')[-1].upper()}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>⏱️ Duration</h4>
                <p>30 seconds</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Process audio
        st.markdown("### 🎵 **Processing Audio**")
        with st.spinner("🎵 Processing audio file..."):
            try:
                features, y, sr = classifier.extract_audio_features(uploaded_file)
                
                if features is not None:
                    st.success("✅ **Audio features extracted successfully!**")
                    
                    # Create spectrogram
                    spectrogram_path = classifier.create_enhanced_spectrogram(y, sr)
                    
                    if spectrogram_path is not None:
                        st.success("✅ **Spectrogram created successfully!**")
                    else:
                        st.warning("⚠️ **Could not create spectrogram, but features were extracted successfully.**")
                else:
                    st.error("❌ **Failed to extract audio features. Please try a different audio file.**")
                    st.info("💡 **Tips:** Make sure your audio file is not corrupted and is in a supported format (WAV, MP3, M4A, FLAC)")
            except Exception as e:
                st.error(f"❌ **Unexpected error during processing:** {e}")
                st.info("💡 **Please try uploading a different audio file or check if the file is not corrupted.**")
        
        # Display results
        if features is not None:
            st.markdown("---")
            st.markdown("## 🎯 **Genre Classification Results**")
            
            # Initialize prediction variables
            tabular_predictions = {}
            cnn_predictions = {}
            
            # Tabular predictions
            if approach in ["Feature-Based (Tabular)", "Both Approaches", "Ensemble Analysis"]:
                st.markdown("### 📊 **Feature-Based Predictions**")
                
                with st.spinner("🤖 Running tabular models..."):
                    tabular_predictions = classifier.predict_genre_tabular(features)
                
                if tabular_predictions:
                    # Display predictions in cards
                    st.markdown("#### 🎵 **Tabular Model Predictions**")
                    cols = st.columns(len(tabular_predictions))
                    
                    for i, (model_name, pred) in enumerate(tabular_predictions.items()):
                        with cols[i]:
                            confidence_color = "🟢" if pred['confidence'] > 0.7 else "🟡" if pred['confidence'] > 0.4 else "🔴"
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h4>🤖 {model_name}</h4>
                                <h3>🎵 {pred['class']}</h3>
                                <h2>{confidence_color} {pred['confidence']:.3f}</h2>
                            </div>
                            """, unsafe_allow_html=True)
            
            # CNN predictions
            if approach in ["CNN-Based (Deep Learning)", "Both Approaches", "Ensemble Analysis"] and spectrogram_path is not None:
                st.markdown("### 🧠 **CNN-Based Predictions**")
                
                with st.spinner("🧠 Running CNN models..."):
                    cnn_predictions = classifier.predict_genre_cnn(spectrogram_path)
                
                if cnn_predictions:
                    # Display predictions in cards
                    st.markdown("#### 🧠 **CNN Model Predictions**")
                    cols = st.columns(len(cnn_predictions))
                    
                    for i, (model_name, pred) in enumerate(cnn_predictions.items()):
                        with cols[i]:
                            confidence_color = "🟢" if pred['confidence'] > 0.7 else "🟡" if pred['confidence'] > 0.4 else "🔴"
                            st.markdown(f"""
                            <div class="cnn-card">
                                <h4>🧠 {model_name}</h4>
                                <h3>🎵 {pred['class']}</h3>
                                <h2>{confidence_color} {pred['confidence']:.3f}</h2>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Enhanced visualizations
            if tabular_predictions or cnn_predictions:
                create_enhanced_visualizations(tabular_predictions, cnn_predictions)
            
            # Display spectrogram
            if spectrogram_path is not None:
                st.markdown("### 🎨 **Generated Spectrogram**")
                st.markdown("""
                <div style="text-align: center; margin: 2rem 0;">
                    <p style="color: #6c757d; font-style: italic;">
                        Visual representation of your audio's frequency content over time
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.image(str(spectrogram_path), caption="🎵 Mel Spectrogram", use_container_width=True)
        
        # Clean up temporary files
        if spectrogram_path and spectrogram_path.exists():
            spectrogram_path.unlink()
    
    else:
        # Enhanced landing page
        st.markdown("## 🚀 **Welcome to Enhanced Music Genre Classification!**")
        
        # Feature highlights
        st.markdown("### ✨ **Key Features**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🤖 Tabular Models</h3>
                <p>6 different ML algorithms for accurate classification</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="cnn-card">
                <h3>🧠 CNN Models</h3>
                <p>Deep learning models for spectrogram analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🎵 10 Genres</h3>
                <p>Classify across diverse music genres</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="cnn-card">
                <h3>📊 Real-time Analysis</h3>
                <p>Instant predictions with confidence scores</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Instructions
        st.markdown("## 🚀 **How to Use**")
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                    padding: 2rem; border-radius: 15px; margin: 2rem 0;">
            <h3>📋 **Step-by-Step Guide**</h3>
            <ol style="font-size: 1.1rem; line-height: 2;">
                <li><strong>📁 Upload Audio:</strong> Use the sidebar to upload a 30-second audio file</li>
                <li><strong>🎯 Select Approach:</strong> Choose between tabular, CNN, both, or ensemble</li>
                <li><strong>🎵 Get Results:</strong> View genre predictions and confidence scores</li>
                <li><strong>📊 Compare Models:</strong> See how different algorithms perform</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Supported genres
        st.markdown("## 🎵 **Supported Genres**")
        genres = ['Blues', 'Classical', 'Country', 'Disco', 'Hip-hop', 
                 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
        
        # Display genres in a grid
        cols = st.columns(5)
        for i, genre in enumerate(genres):
            with cols[i % 5]:
                st.markdown(f'<div style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.5rem 1rem; border-radius: 25px; margin: 0.25rem; font-weight: 500; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);">🎵 {genre}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
