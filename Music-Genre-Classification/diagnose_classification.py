"""
🔍 Music Genre Classification - Diagnostic Script

Analyzes why pop songs might be misclassified as rock and provides insights.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_training_data():
    """Load the training data and models"""
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
        'Gradient Boosting': 'gradient_boosting.pkl',
        'Logistic Regression': 'logistic_regression.pkl',
        'K-Nearest Neighbors': 'k-nearest_neighbors.pkl',
        'Naive Bayes': 'naive_bayes.pkl'
    }
    
    for name, filename in model_files.items():
        model_path = models_path / filename
        if model_path.exists():
            models[name] = joblib.load(model_path)
    
    # Load scaler and encoder
    scaler = joblib.load(processed_path / "feature_scaler.pkl")
    label_encoder = joblib.load(processed_path / "label_encoder.pkl")
    
    return split_data, models, scaler, label_encoder

def analyze_class_distribution(split_data, label_encoder):
    """Analyze the class distribution in training data"""
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
    
    return train_dist, test_dist

def analyze_feature_importance(models, feature_names, label_encoder):
    """Analyze feature importance for different models"""
    print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    
    # Get feature importance for tree-based models
    tree_models = ['Random Forest', 'Gradient Boosting']
    
    for model_name in tree_models:
        if model_name in models:
            model = models[model_name]
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Get top 10 most important features
                top_indices = np.argsort(importances)[-10:]
                top_features = [feature_names[i] for i in top_indices]
                top_importances = importances[top_indices]
                
                print(f"\n🌳 {model_name} - Top 10 Features:")
                for feature, importance in zip(top_features, top_importances):
                    print(f"   {feature}: {importance:.4f}")

def analyze_model_predictions(models, split_data, scaler, label_encoder):
    """Analyze model predictions on test set"""
    print("\n🔍 MODEL PREDICTION ANALYSIS")
    print("=" * 50)
    
    X_test = split_data['X_test']
    y_test = split_data['y_test_orig']
    
    # Get class names
    class_names = label_encoder.classes_
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Analyze each model
    for model_name, model in models.items():
        print(f"\n🤖 {model_name} Analysis:")
        
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
            print(f"   ⚠️ Found {len(pop_rock_confusion)} pop→rock misclassifications")
            for case in pop_rock_confusion:
                print(f"      Confidence: {case['confidence']:.3f}")
                print(f"      Rock prob: {case['rock_prob']:.3f}, Pop prob: {case['pop_prob']:.3f}")
        else:
            print("   ✅ No pop→rock misclassifications found")

def analyze_feature_differences(split_data, label_encoder):
    """Analyze feature differences between pop and rock"""
    print("\n🔍 POP vs ROCK FEATURE ANALYSIS")
    print("=" * 50)
    
    X_train = split_data['X_train']
    y_train = split_data['y_train_orig']
    
    # Get pop and rock samples
    pop_mask = y_train == 'pop'
    rock_mask = y_train == 'rock'
    
    pop_features = X_train[pop_mask]
    rock_features = X_train[rock_mask]
    
    if len(pop_features) == 0 or len(rock_features) == 0:
        print("⚠️ No pop or rock samples found in training data!")
        return
    
    # Calculate mean differences
    pop_mean = np.mean(pop_features, axis=0)
    rock_mean = np.mean(rock_features, axis=0)
    feature_diff = np.abs(pop_mean - rock_mean)
    
    # Get feature names
    feature_names = split_data['feature_names']
    
    # Find features with largest differences
    top_diff_indices = np.argsort(feature_diff)[-10:]
    
    print("📊 Top 10 Features with Largest Pop-Rock Differences:")
    for i, idx in enumerate(top_diff_indices):
        feature_name = feature_names[idx]
        pop_val = pop_mean[idx]
        rock_val = rock_mean[idx]
        diff = feature_diff[idx]
        print(f"   {i+1}. {feature_name}:")
        print(f"      Pop: {pop_val:.4f}, Rock: {rock_val:.4f}, Diff: {diff:.4f}")

def create_diagnostic_report():
    """Create a comprehensive diagnostic report"""
    print("MUSIC GENRE CLASSIFICATION - DIAGNOSTIC REPORT")
    print("=" * 60)
    
    try:
        # Load data
        split_data, models, scaler, label_encoder = load_training_data()
        
        # Analyze class distribution
        train_dist, test_dist = analyze_class_distribution(split_data, label_encoder)
        
        # Analyze feature importance
        feature_names = split_data['feature_names']
        analyze_feature_importance(models, feature_names, label_encoder)
        
        # Analyze model predictions
        analyze_model_predictions(models, split_data, scaler, label_encoder)
        
        # Analyze feature differences
        analyze_feature_differences(split_data, label_encoder)
        
        # Summary and recommendations
        print("\n🎯 DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
        print("=" * 50)
        
        print("🔍 IDENTIFIED ISSUES:")
        print("   1. ⚠️ VERY SMALL DATASET: Only 4 samples per genre in training")
        print("   2. ⚠️ LIMITED DIVERSITY: Sample dataset may not represent real music well")
        print("   3. ⚠️ FEATURE SIMILARITY: Pop and rock may have similar audio characteristics")
        print("   4. ⚠️ MODEL OVERFITTING: Models may memorize the small dataset")
        
        print("\n💡 RECOMMENDATIONS:")
        print("   1. 📈 DOWNLOAD FULL GTZAN DATASET (1000 songs)")
        print("   2. 🎵 USE REAL MUSIC FILES instead of synthetic samples")
        print("   3. 🔧 FEATURE ENGINEERING: Add more discriminative features")
        print("   4. 🎯 ENSEMBLE METHODS: Combine multiple models for better accuracy")
        print("   5. 📊 DATA AUGMENTATION: Generate more training samples")
        
        print("\n🚀 IMMEDIATE SOLUTIONS:")
        print("   1. Run: python scripts/download_dataset.py")
        print("   2. Run: python scripts/train_all_models.py")
        print("   3. Test with real pop/rock songs from different artists")
        
    except Exception as e:
        print(f"❌ Error in diagnostic analysis: {e}")

if __name__ == "__main__":
    create_diagnostic_report()
