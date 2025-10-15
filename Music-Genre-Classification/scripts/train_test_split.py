"""
Music Genre Classification - Train/Test Split Script

Creates proper train/test splits for both feature-based and image-based approaches.
Implements stratified splitting to ensure balanced representation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json

def load_feature_data(data_path):
    """Load extracted audio features"""
    processed_path = data_path / "processed"
    
    # Try to load from pickle first (faster)
    pickle_path = processed_path / "audio_features.pkl"
    if pickle_path.exists():
        print("[INFO] Loading features from pickle...")
        data = joblib.load(pickle_path)
        X = data['features']
        y = data['labels']
        feature_names = data['feature_names']
        file_paths = data['file_paths']
    else:
        # Load from CSV
        csv_path = processed_path / "audio_features.csv"
        if not csv_path.exists():
            print("[ERROR] Feature data not found!")
            print("Please run: python scripts/feature_extraction.py")
            return None
        
        print("[INFO] Loading features from CSV...")
        df = pd.read_csv(csv_path)
        X = df.drop(['genre', 'file_path'], axis=1).values
        y = df['genre'].values
        feature_names = df.drop(['genre', 'file_path'], axis=1).columns.tolist()
        file_paths = df['file_path'].values
    
    print(f"   • Samples: {len(X)}")
    print(f"   • Features: {X.shape[1]}")
    print(f"   • Genres: {len(np.unique(y))}")
    
    return X, y, feature_names, file_paths

def create_stratified_split(X, y, test_size=0.2, random_state=42):
    """Create stratified train/test split"""
    print(f"\n[INFO] Creating stratified train/test split (test_size={test_size})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"   • Train samples: {len(X_train)}")
    print(f"   • Test samples: {len(X_test)}")
    
    # Check class distribution
    train_dist = pd.Series(y_train).value_counts().sort_index()
    test_dist = pd.Series(y_test).value_counts().sort_index()
    
    print("\n[INFO] Class distribution:")
    print("Genre".ljust(12) + "Train".ljust(8) + "Test".ljust(8) + "Total")
    print("-" * 40)
    for genre in sorted(np.unique(y)):
        train_count = train_dist.get(genre, 0)
        test_count = test_dist.get(genre, 0)
        total_count = train_count + test_count
        print(f"{genre.ljust(12)}{str(train_count).ljust(8)}{str(test_count).ljust(8)}{total_count}")
    
    return X_train, X_test, y_train, y_test

def standardize_features(X_train, X_test):
    """Standardize features using StandardScaler"""
    print("\n[STANDARDIZING] Features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("   • Features standardized (mean=0, std=1)")
    
    return X_train_scaled, X_test_scaled, scaler

def encode_labels(y_train, y_test):
    """Encode string labels to integers"""
    print("\n[INFO] Encoding labels...")
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"   • Classes: {len(label_encoder.classes_)}")
    print(f"   • Labels: {label_encoder.classes_}")
    
    return y_train_encoded, y_test_encoded, label_encoder

def save_split_data(data_path, X_train, X_test, y_train, y_test, 
                   y_train_orig, y_test_orig, feature_names, scaler, label_encoder):
    """Save train/test split data"""
    processed_path = data_path / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Save split data
    split_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_orig': y_train_orig,
        'y_test_orig': y_test_orig,
        'feature_names': feature_names
    }
    
    split_path = processed_path / "train_test_split.pkl"
    joblib.dump(split_data, split_path)
    print(f"[SAVED] Split data to: {split_path}")
    
    # Save scaler and encoder
    scaler_path = processed_path / "feature_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"[SAVED] Scaler to: {scaler_path}")
    
    encoder_path = processed_path / "label_encoder.pkl"
    joblib.dump(label_encoder, encoder_path)
    print(f"[SAVED] Label encoder to: {encoder_path}")
    
    # Save metadata
    metadata = {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': X_train.shape[1],
        'n_classes': len(label_encoder.classes_),
        'classes': label_encoder.classes_.tolist(),
        'feature_names': feature_names,
        'train_class_distribution': pd.Series(y_train_orig).value_counts().to_dict(),
        'test_class_distribution': pd.Series(y_test_orig).value_counts().to_dict()
    }
    
    metadata_path = processed_path / "split_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[SUCCESS] Saved metadata to: {metadata_path}")

def create_cv_splits(X, y, n_splits=3, random_state=42):
    """Create cross-validation splits"""
    print(f"\n[INFO] Creating {n_splits}-fold cross-validation splits...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_splits = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        cv_splits.append({
            'fold': fold,
            'X_train': X_train_fold,
            'X_val': X_val_fold,
            'y_train': y_train_fold,
            'y_val': y_val_fold
        })
        
        print(f"   • Fold {fold+1}: Train={len(X_train_fold)}, Val={len(X_val_fold)}")
    
    return cv_splits

def prepare_spectrogram_data(data_path):
    """Prepare spectrogram data for CNN training"""
    print("\n[INFO] Preparing spectrogram data...")
    
    spectrogram_path = data_path / "spectrograms"
    if not spectrogram_path.exists():
        print("[ERROR] Spectrogram data not found!")
        print("Please run: python scripts/create_spectrograms.py")
        return None
    
    # Get all spectrogram files
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    image_files = []
    labels = []
    
    for genre in genres:
        genre_path = spectrogram_path / "mel" / genre
        if genre_path.exists():
            files = list(genre_path.glob("*.png"))
            image_files.extend(files)
            labels.extend([genre] * len(files))
            print(f"   • {genre}: {len(files)} spectrograms")
    
    if not image_files:
        print("[ERROR] No spectrogram files found!")
        return None
    
    # Create stratified split for images
    X_img_train, X_img_test, y_img_train, y_img_test = train_test_split(
        image_files, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    print(f"   • Train images: {len(X_img_train)}")
    print(f"   • Test images: {len(X_img_test)}")
    
    # Save image split data
    img_split_data = {
        'X_train': X_img_train,
        'X_test': X_img_test,
        'y_train': y_img_train,
        'y_test': y_img_test
    }
    
    processed_path = data_path / "processed"
    img_split_path = processed_path / "image_train_test_split.pkl"
    joblib.dump(img_split_data, img_split_path)
    print(f"[SAVED] Image split data to: {img_split_path}")
    
    return img_split_data

def main():
    """Main function"""
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data"
    
    print("Music Genre Classification - Train/Test Split")
    print("=" * 60)
    
    # Load feature data
    feature_data = load_feature_data(data_path)
    if feature_data is None:
        return
    
    X, y, feature_names, file_paths = feature_data
    
    # Create stratified split
    X_train, X_test, y_train, y_test = create_stratified_split(X, y)
    
    # Standardize features
    X_train_scaled, X_test_scaled, scaler = standardize_features(X_train, X_test)
    
    # Encode labels
    y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)
    
    # Save split data
    save_split_data(
        data_path, X_train_scaled, X_test_scaled, 
        y_train_encoded, y_test_encoded,
        y_train, y_test, feature_names, scaler, label_encoder
    )
    
    # Create cross-validation splits
    cv_splits = create_cv_splits(X_train_scaled, y_train_encoded)
    cv_path = data_path / "processed" / "cv_splits.pkl"
    joblib.dump(cv_splits, cv_path)
    print(f"[SUCCESS] Saved CV splits to: {cv_path}")
    
    # Prepare spectrogram data
    img_data = prepare_spectrogram_data(data_path)
    
    print("\n[NEXT] Next Steps:")
    print("   1. Run: python scripts/train_tabular_models.py")
    print("   2. Run: python scripts/train_cnn_models.py")
    print("   3. Run: python scripts/evaluate_models.py")

if __name__ == "__main__":
    main()
