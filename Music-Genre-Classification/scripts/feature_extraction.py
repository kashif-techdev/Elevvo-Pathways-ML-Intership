"""
Music Genre Classification - Feature Extraction Script

Extracts audio features using librosa for the feature-based approach.
Implements MFCC, spectral features, and other audio characteristics.
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from pathlib import Path
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    """Extract comprehensive audio features for genre classification"""
    
    def __init__(self, sample_rate=22050, duration=30):
        self.sample_rate = sample_rate
        self.duration = duration
        self.feature_names = []
        
    def extract_mfcc(self, y, sr, n_mfcc=20):
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        return np.hstack([mfcc_mean, mfcc_std])
    
    def extract_spectral_features(self, y, sr):
        """Extract spectral features"""
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroids)
        spectral_centroid_std = np.std(spectral_centroids)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_bandwidth_std = np.std(spectral_bandwidth)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        spectral_rolloff_std = np.std(spectral_rolloff)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        return np.array([
            spectral_centroid_mean, spectral_centroid_std,
            spectral_bandwidth_mean, spectral_bandwidth_std,
            spectral_rolloff_mean, spectral_rolloff_std,
            zcr_mean, zcr_std
        ])
    
    def extract_chroma_features(self, y, sr):
        """Extract chroma features"""
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        return np.hstack([chroma_mean, chroma_std])
    
    def extract_rhythm_features(self, y, sr):
        """Extract rhythm and tempo features"""
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Rhythm features
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        if len(onset_times) > 1:
            onset_intervals = np.diff(onset_times)
            rhythm_mean = np.mean(onset_intervals)
            rhythm_std = np.std(onset_intervals)
        else:
            rhythm_mean = 0
            rhythm_std = 0
        
        return np.array([tempo, rhythm_mean, rhythm_std])
    
    def extract_tonnetz_features(self, y, sr):
        """Extract tonnetz features"""
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        tonnetz_std = np.std(tonnetz, axis=1)
        return np.hstack([tonnetz_mean, tonnetz_std])
    
    def extract_features(self, file_path):
        """Extract all features from audio file"""
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # Extract different feature groups
            mfcc_features = self.extract_mfcc(y, sr)
            spectral_features = self.extract_spectral_features(y, sr)
            chroma_features = self.extract_chroma_features(y, sr)
            rhythm_features = self.extract_rhythm_features(y, sr)
            tonnetz_features = self.extract_tonnetz_features(y, sr)
            
            # Combine all features
            features = np.hstack([
                mfcc_features,
                spectral_features,
                chroma_features,
                rhythm_features,
                tonnetz_features
            ])
            
            return features
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def get_feature_names(self):
        """Get names of all extracted features"""
        if not self.feature_names:
            # MFCC features (20 mean + 20 std = 40)
            mfcc_names = [f'mfcc_{i}_mean' for i in range(20)] + [f'mfcc_{i}_std' for i in range(20)]
            
            # Spectral features (8)
            spectral_names = [
                'spectral_centroid_mean', 'spectral_centroid_std',
                'spectral_bandwidth_mean', 'spectral_bandwidth_std',
                'spectral_rolloff_mean', 'spectral_rolloff_std',
                'zcr_mean', 'zcr_std'
            ]
            
            # Chroma features (12 mean + 12 std = 24)
            chroma_names = [f'chroma_{i}_mean' for i in range(12)] + [f'chroma_{i}_std' for i in range(12)]
            
            # Rhythm features (3)
            rhythm_names = ['tempo', 'rhythm_mean', 'rhythm_std']
            
            # Tonnetz features (6 mean + 6 std = 12)
            tonnetz_names = [f'tonnetz_{i}_mean' for i in range(6)] + [f'tonnetz_{i}_std' for i in range(6)]
            
            self.feature_names = mfcc_names + spectral_names + chroma_names + rhythm_names + tonnetz_names
        
        return self.feature_names

def process_dataset(data_path, output_path):
    """Process entire dataset and extract features"""
    print("Music Genre Classification - Feature Extraction")
    print("=" * 60)
    
    # Initialize feature extractor
    extractor = AudioFeatureExtractor()
    
    # Get all audio files
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    all_features = []
    all_labels = []
    all_files = []
    
    print(f"[INFO] Processing {len(genres)} genres...")
    
    for genre in genres:
        genre_path = data_path / "raw" / genre
        if not genre_path.exists():
            print(f"[WARNING] Genre folder not found: {genre_path}")
            continue
        
        audio_files = list(genre_path.glob("*.wav")) + list(genre_path.glob("*.au"))
        print(f"[GENRE] {genre}: {len(audio_files)} files")
        
        for file_path in tqdm(audio_files, desc=f"Processing {genre}"):
            features = extractor.extract_features(file_path)
            if features is not None:
                all_features.append(features)
                all_labels.append(genre)
                all_files.append(str(file_path))
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\n[SUCCESS] Feature Extraction Complete!")
    print(f"   • Total samples: {len(X)}")
    print(f"   • Feature dimensions: {X.shape[1]}")
    print(f"   • Genres: {len(np.unique(y))}")
    
    # Create DataFrame
    feature_names = extractor.get_feature_names()
    df = pd.DataFrame(X, columns=feature_names)
    df['genre'] = y
    df['file_path'] = all_files
    
    # Save features
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_path = output_path / "audio_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved features to: {csv_path}")
    
    # Save as pickle for faster loading
    pickle_path = output_path / "audio_features.pkl"
    joblib.dump({
        'features': X,
        'labels': y,
        'feature_names': feature_names,
        'file_paths': all_files
    }, pickle_path)
    print(f"💾 Saved features to: {pickle_path}")
    
    # Save feature statistics
    stats = {
        'total_samples': len(X),
        'feature_dimensions': X.shape[1],
        'genres': list(np.unique(y)),
        'genre_counts': dict(pd.Series(y).value_counts()),
        'feature_names': feature_names
    }
    
    stats_path = output_path / "feature_stats.json"
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[SUCCESS] Saved statistics to: {stats_path}")
    
    return df, X, y

def main():
    """Main function"""
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data"
    output_path = data_path / "processed"
    
    # Check if raw data exists
    raw_path = data_path / "raw"
    if not raw_path.exists():
        print("[ERROR] Raw data folder not found!")
        print("Please run: python scripts/download_dataset.py")
        return
    
    # Process dataset
    df, X, y = process_dataset(data_path, output_path)
    
    print("\n[NEXT] Next Steps:")
    print("   1. Run: python scripts/train_test_split.py")
    print("   2. Run: python scripts/train_tabular_models.py")
    print("   3. Run: python scripts/create_spectrograms.py")

if __name__ == "__main__":
    main()
