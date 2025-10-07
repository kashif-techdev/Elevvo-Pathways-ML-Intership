"""
🎵 Music Genre Classification - Tabular Models Training Script

Trains classical machine learning models for feature-based approach.
Implements Random Forest, SVM, and Gradient Boosting classifiers.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TabularModelTrainer:
    """Train and evaluate tabular machine learning models"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load preprocessed data"""
        processed_path = self.data_path / "processed"
        
        # Load split data
        split_path = processed_path / "train_test_split.pkl"
        if not split_path.exists():
            print("❌ Train/test split not found!")
            print("Please run: python scripts/train_test_split.py")
            return None
        
        print("📊 Loading train/test split data...")
        split_data = joblib.load(split_path)
        
        self.X_train = split_data['X_train']
        self.X_test = split_data['X_test']
        self.y_train = split_data['y_train']
        self.y_test = split_data['y_test']
        self.y_train_orig = split_data['y_train_orig']
        self.y_test_orig = split_data['y_test_orig']
        self.feature_names = split_data['feature_names']
        
        # Load label encoder
        encoder_path = processed_path / "label_encoder.pkl"
        self.label_encoder = joblib.load(encoder_path)
        
        print(f"   • Train samples: {len(self.X_train)}")
        print(f"   • Test samples: {len(self.X_test)}")
        print(f"   • Features: {self.X_train.shape[1]}")
        print(f"   • Classes: {len(self.label_encoder.classes_)}")
        
        return True
    
    def define_models(self):
        """Define models to train"""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                random_state=42,
                probability=True
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5
            ),
            'Naive Bayes': GaussianNB()
        }
        
        print(f"🤖 Defined {len(self.models)} models for training")
    
    def train_models(self):
        """Train all models"""
        print("\n🚀 Training models...")
        
        for name, model in tqdm(self.models.items(), desc="Training"):
            print(f"\n🎯 Training {name}...")
            
            try:
                # Train model
                model.fit(self.X_train, self.y_train)
                
                # Make predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'classification_report': classification_report(
                        self.y_test, y_pred, 
                        target_names=self.label_encoder.classes_,
                        output_dict=True
                    )
                }
                
                print(f"   ✅ Accuracy: {accuracy:.4f}")
                print(f"   ✅ CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {e}")
                continue
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for best models"""
        print("\n🔧 Performing hyperparameter tuning...")
        
        # Random Forest tuning
        print("🎯 Tuning Random Forest...")
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            rf_params,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        rf_grid.fit(self.X_train, self.y_train)
        
        print(f"   Best RF params: {rf_grid.best_params_}")
        print(f"   Best RF score: {rf_grid.best_score_:.4f}")
        
        # SVM tuning
        print("🎯 Tuning SVM...")
        svm_params = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.001, 0.01]
        }
        
        svm_grid = GridSearchCV(
            SVC(random_state=42, probability=True),
            svm_params,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        svm_grid.fit(self.X_train, self.y_train)
        
        print(f"   Best SVM params: {svm_grid.best_params_}")
        print(f"   Best SVM score: {svm_grid.best_score_:.4f}")
        
        # Update models with best parameters
        self.models['Random Forest'] = rf_grid.best_estimator_
        self.models['SVM'] = svm_grid.best_estimator_
        
        return rf_grid.best_estimator_, svm_grid.best_estimator_
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n📊 Model Evaluation Results:")
        print("=" * 80)
        print(f"{'Model':<20} {'Accuracy':<10} {'CV Mean':<10} {'CV Std':<10} {'F1-Score':<10}")
        print("-" * 80)
        
        for name, result in self.results.items():
            f1_score = result['classification_report']['macro avg']['f1-score']
            print(f"{name:<20} {result['accuracy']:<10.4f} {result['cv_mean']:<10.4f} "
                  f"{result['cv_std']:<10.4f} {f1_score:<10.4f}")
        
        # Find best model
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['accuracy'])
        best_accuracy = self.results[best_model_name]['accuracy']
        
        print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        return best_model_name
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        print("\n📊 Creating confusion matrices...")
        
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (name, result) in enumerate(self.results.items()):
            if i >= 6:  # Limit to 6 plots
                break
                
            cm = confusion_matrix(self.y_test, result['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_,
                       ax=axes[i])
            axes[i].set_title(f'{name}\nAccuracy: {result["accuracy"]:.4f}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(len(self.results), 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plots_path = self.data_path / "processed" / "confusion_matrices.png"
        plt.savefig(plots_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved confusion matrices to: {plots_path}")
        plt.show()
    
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        print("\n📊 Creating feature importance plots...")
        
        tree_models = ['Random Forest', 'Gradient Boosting']
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        for i, model_name in enumerate(tree_models):
            if model_name in self.results:
                model = self.results[model_name]['model']
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1][:20]  # Top 20 features
                    
                    axes[i].bar(range(20), importances[indices])
                    axes[i].set_title(f'{model_name} - Top 20 Feature Importance')
                    axes[i].set_xlabel('Features')
                    axes[i].set_ylabel('Importance')
                    axes[i].set_xticks(range(20))
                    axes[i].set_xticklabels([self.feature_names[j] for j in indices], 
                                          rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        plots_path = self.data_path / "processed" / "feature_importance.png"
        plt.savefig(plots_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved feature importance to: {plots_path}")
        plt.show()
    
    def save_models(self):
        """Save trained models"""
        print("\n💾 Saving trained models...")
        
        models_path = self.data_path / "models"
        models_path.mkdir(parents=True, exist_ok=True)
        
        for name, result in self.results.items():
            model_path = models_path / f"{name.lower().replace(' ', '_')}.pkl"
            joblib.dump(result['model'], model_path)
            print(f"   💾 Saved {name} to: {model_path}")
        
        # Save results summary
        results_summary = {}
        for name, result in self.results.items():
            results_summary[name] = {
                'accuracy': result['accuracy'],
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std'],
                'f1_score': result['classification_report']['macro avg']['f1-score']
            }
        
        results_path = models_path / "tabular_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"💾 Saved results to: {results_path}")

def main():
    """Main function"""
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data"
    
    print("🎵 Music Genre Classification - Tabular Models Training")
    print("=" * 70)
    
    # Initialize trainer
    trainer = TabularModelTrainer(data_path)
    
    # Load data
    if not trainer.load_data():
        return
    
    # Define models
    trainer.define_models()
    
    # Train models
    trainer.train_models()
    
    # Hyperparameter tuning
    trainer.hyperparameter_tuning()
    
    # Retrain with best parameters
    trainer.train_models()
    
    # Evaluate models
    best_model = trainer.evaluate_models()
    
    # Create visualizations
    trainer.plot_confusion_matrices()
    trainer.plot_feature_importance()
    
    # Save models
    trainer.save_models()
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Run: python scripts/train_cnn_models.py")
    print(f"   2. Run: python scripts/evaluate_models.py")
    print(f"   3. Run: streamlit run app.py")

if __name__ == "__main__":
    main()
