"""
Music Genre Classification - Simple Diagnostic Script
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

def main():
    print("MUSIC GENRE CLASSIFICATION - DIAGNOSTIC REPORT")
    print("=" * 60)
    
    try:
        # Load data
        data_path = Path("data")
        processed_path = data_path / "processed"
        models_path = data_path / "models"
        
        # Load split data
        split_path = processed_path / "train_test_split.pkl"
        split_data = joblib.load(split_path)
        
        # Load models
        models = {}
        model_files = {
            'Random Forest': 'random_forest.pkl',
            'SVM': 'svm.pkl',
            'Gradient Boosting': 'gradient_boosting.pkl'
        }
        
        for name, filename in model_files.items():
            model_path = models_path / filename
            if model_path.exists():
                models[name] = joblib.load(model_path)
        
        # Load scaler and encoder
        scaler = joblib.load(processed_path / "feature_scaler.pkl")
        label_encoder = joblib.load(processed_path / "label_encoder.pkl")
        
        print("CLASS DISTRIBUTION ANALYSIS")
        print("=" * 50)
        
        # Get class names
        class_names = label_encoder.classes_
        
        # Training distribution
        train_labels = split_data['y_train_orig']
        train_dist = pd.Series(train_labels).value_counts().sort_index()
        
        print("Training Set Distribution:")
        for genre in class_names:
            count = train_dist.get(genre, 0)
            print(f"   {genre}: {count} samples")
        
        # Test distribution
        test_labels = split_data['y_test_orig']
        test_dist = pd.Series(test_labels).value_counts().sort_index()
        
        print("\nTest Set Distribution:")
        for genre in class_names:
            count = test_dist.get(genre, 0)
            print(f"   {genre}: {count} samples")
        
        # Calculate imbalance
        print("\nDATA IMBALANCE ANALYSIS:")
        total_train = len(train_labels)
        for genre in class_names:
            count = train_dist.get(genre, 0)
            percentage = (count / total_train) * 100
            print(f"   {genre}: {percentage:.1f}%")
        
        # Analyze model predictions
        print("\nMODEL PREDICTION ANALYSIS")
        print("=" * 50)
        
        X_test = split_data['X_test']
        y_test = split_data['y_test_orig']
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Analyze each model
        for model_name, model in models.items():
            print(f"\n{model_name} Analysis:")
            
            # Get predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_labels = label_encoder.inverse_transform(y_pred)
            
            # Get prediction probabilities
            y_proba = model.predict_proba(X_test_scaled)
            
            # Find pop vs rock confusion
            pop_rock_confusion = []
            for i, (true_label, pred_label) in enumerate(zip(y_test, y_pred_labels)):
                if true_label == 'pop' and pred_label == 'rock':
                    pop_rock_confusion.append({
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'confidence': y_proba[i][y_pred[i]],
                        'rock_prob': y_proba[i][label_encoder.transform(['rock'])[0]],
                        'pop_prob': y_proba[i][label_encoder.transform(['pop'])[0]]
                    })
            
            if pop_rock_confusion:
                print(f"   Found {len(pop_rock_confusion)} pop->rock misclassifications")
                for case in pop_rock_confusion:
                    print(f"      Confidence: {case['confidence']:.3f}")
                    print(f"      Rock prob: {case['rock_prob']:.3f}, Pop prob: {case['pop_prob']:.3f}")
            else:
                print("   No pop->rock misclassifications found")
        
        # Summary and recommendations
        print("\nDIAGNOSTIC SUMMARY & RECOMMENDATIONS")
        print("=" * 50)
        
        print("IDENTIFIED ISSUES:")
        print("   1. VERY SMALL DATASET: Only 4 samples per genre in training")
        print("   2. LIMITED DIVERSITY: Sample dataset may not represent real music well")
        print("   3. FEATURE SIMILARITY: Pop and rock may have similar audio characteristics")
        print("   4. MODEL OVERFITTING: Models may memorize the small dataset")
        
        print("\nRECOMMENDATIONS:")
        print("   1. DOWNLOAD FULL GTZAN DATASET (1000 songs)")
        print("   2. USE REAL MUSIC FILES instead of synthetic samples")
        print("   3. FEATURE ENGINEERING: Add more discriminative features")
        print("   4. ENSEMBLE METHODS: Combine multiple models for better accuracy")
        print("   5. DATA AUGMENTATION: Generate more training samples")
        
        print("\nIMMEDIATE SOLUTIONS:")
        print("   1. Run: python scripts/download_dataset.py")
        print("   2. Run: python scripts/train_all_models.py")
        print("   3. Test with real pop/rock songs from different artists")
        
    except Exception as e:
        print(f"Error in diagnostic analysis: {e}")

if __name__ == "__main__":
    main()
