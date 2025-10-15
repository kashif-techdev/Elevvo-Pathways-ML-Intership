"""
[MUSIC] Music Genre Classification - Model Evaluation Script

Comprehensive evaluation of all trained models.
Generates detailed performance metrics and visualizations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import cross_val_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation and comparison"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.processed_path = data_path / "processed"
        self.models_path = data_path / "models"
        self.results = {}
        
    def load_data(self):
        """Load test data and models"""
        print("[CHART] Loading test data and models...")
        
        # Load split data
        split_path = self.processed_path / "train_test_split.pkl"
        if not split_path.exists():
            print("[ERROR] Train/test split not found!")
            return False
        
        split_data = joblib.load(split_path)
        self.X_test = split_data['X_test']
        self.y_test = split_data['y_test']
        self.y_test_orig = split_data['y_test_orig']
        
        # Load label encoder
        encoder_path = self.processed_path / "label_encoder.pkl"
        self.label_encoder = joblib.load(encoder_path)
        
        print(f"   • Test samples: {len(self.X_test)}")
        print(f"   • Classes: {len(self.label_encoder.classes_)}")
        
        return True
    
    def load_tabular_models(self):
        """Load tabular models"""
        print("\n🤖 Loading tabular models...")
        
        self.tabular_models = {}
        tabular_model_files = {
            'Random Forest': 'random_forest.pkl',
            'SVM': 'svm.pkl',
            'Gradient Boosting': 'gradient_boosting.pkl',
            'Logistic Regression': 'logistic_regression.pkl',
            'K-Nearest Neighbors': 'k-nearest_neighbors.pkl',
            'Naive Bayes': 'naive_bayes.pkl'
        }
        
        for name, filename in tabular_model_files.items():
            model_path = self.models_path / filename
            if model_path.exists():
                self.tabular_models[name] = joblib.load(model_path)
                print(f"   [OK] Loaded {name}")
            else:
                print(f"   [WARNING]  {name} not found")
        
        print(f"   • Loaded {len(self.tabular_models)} tabular models")
    
    def evaluate_tabular_models(self):
        """Evaluate tabular models"""
        print("\n[CHART] Evaluating tabular models...")
        
        for name, model in self.tabular_models.items():
            print(f"\n[TARGET] Evaluating {name}...")
            
            try:
                # Make predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted')
                recall = recall_score(self.y_test, y_pred, average='weighted')
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                
                # Cross-validation score
                cv_scores = cross_val_score(model, self.X_test, self.y_test, cv=5)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Store results
                self.results[name] = {
                    'type': 'tabular',
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                    'classification_report': classification_report(
                        self.y_test, y_pred, 
                        target_names=self.label_encoder.classes_,
                        output_dict=True
                    )
                }
                
                print(f"   [OK] Accuracy: {accuracy:.4f}")
                print(f"   [OK] F1-Score: {f1:.4f}")
                print(f"   [OK] CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
                
            except Exception as e:
                print(f"   [ERROR] Error evaluating {name}: {e}")
                continue
    
    def create_performance_comparison(self):
        """Create performance comparison visualization"""
        print("\n[CHART] Creating performance comparison...")
        
        # Prepare data for visualization
        model_names = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        colors = px.colors.qualitative.Set3[:len(model_names)]
        
        for i, metric in enumerate(metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            values = [self.results[name][metric] for name in model_names]
            
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=values,
                    name=metric,
                    marker_color=colors,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Model Performance Comparison",
            height=800,
            showlegend=False
        )
        
        # Save plot
        plots_path = self.processed_path / "performance_comparison.png"
        fig.write_image(str(plots_path))
        print(f"[SAVED] Saved performance comparison to: {plots_path}")
        
        return fig
    
    def create_confusion_matrices(self):
        """Create confusion matrices for all models"""
        print("\n[CHART] Creating confusion matrices...")
        
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (name, result) in enumerate(self.results.items()):
            if i >= 6:  # Limit to 6 plots
                break
                
            cm = result['confusion_matrix']
            
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
        plots_path = self.processed_path / "confusion_matrices.png"
        plt.savefig(plots_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Saved confusion matrices to: {plots_path}")
        plt.show()
    
    def create_roc_curves(self):
        """Create ROC curves for models with probabilities"""
        print("\n[CHART] Creating ROC curves...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for name, result in self.results.items():
            if result['probabilities'] is not None:
                try:
                    # Calculate ROC AUC for each class
                    y_test_binary = np.eye(len(self.label_encoder.classes_))[self.y_test]
                    roc_auc = roc_auc_score(y_test_binary, result['probabilities'], multi_class='ovr')
                    
                    # Plot ROC curve for each class
                    for i, class_name in enumerate(self.label_encoder.classes_):
                        fpr, tpr, _ = roc_curve(y_test_binary[:, i], result['probabilities'][:, i])
                        ax.plot(fpr, tpr, label=f'{name} - {class_name}', alpha=0.7)
                    
                except Exception as e:
                    print(f"   [WARNING]  Could not create ROC curve for {name}: {e}")
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plots_path = self.processed_path / "roc_curves.png"
        plt.savefig(plots_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Saved ROC curves to: {plots_path}")
        plt.show()
    
    def create_class_performance(self):
        """Create per-class performance analysis"""
        print("\n[CHART] Creating per-class performance analysis...")
        
        # Prepare data
        class_names = self.label_encoder.classes_
        model_names = list(self.results.keys())
        
        # Create performance matrix
        performance_data = []
        
        for model_name in model_names:
            result = self.results[model_name]
            report = result['classification_report']
            
            for class_name in class_names:
                if class_name in report:
                    performance_data.append({
                        'Model': model_name,
                        'Class': class_name,
                        'Precision': report[class_name]['precision'],
                        'Recall': report[class_name]['recall'],
                        'F1-Score': report[class_name]['f1-score']
                    })
        
        df = pd.DataFrame(performance_data)
        
        # Create heatmap
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for i, metric in enumerate(['Precision', 'Recall', 'F1-Score']):
            pivot_df = df.pivot(index='Model', columns='Class', values=metric)
            
            sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd',
                       ax=axes[i], cbar_kws={'label': metric})
            axes[i].set_title(f'{metric} by Class and Model')
            axes[i].set_xlabel('Class')
            axes[i].set_ylabel('Model')
        
        plt.tight_layout()
        
        # Save plot
        plots_path = self.processed_path / "class_performance.png"
        plt.savefig(plots_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Saved class performance to: {plots_path}")
        plt.show()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n[CHART] Generating summary report...")
        
        # Create summary DataFrame
        summary_data = []
        
        for name, result in self.results.items():
            summary_data.append({
                'Model': name,
                'Type': result['type'],
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score'],
                'CV Mean': result['cv_mean'],
                'CV Std': result['cv_std']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Accuracy', ascending=False)
        
        # Save summary
        summary_path = self.processed_path / "evaluation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[SAVED] Saved summary to: {summary_path}")
        
        # Print summary
        print("\n[CHART] Evaluation Summary:")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        
        # Find best model
        best_model = summary_df.iloc[0]
        print(f"\n🏆 Best Model: {best_model['Model']} (Accuracy: {best_model['Accuracy']:.4f})")
        
        # Save detailed results
        detailed_results = {}
        for name, result in self.results.items():
            detailed_results[name] = {
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1_score': result['f1_score'],
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std']
            }
        
        results_path = self.processed_path / "detailed_evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"[SAVED] Saved detailed results to: {results_path}")
        
        return summary_df

def main():
    """Main function"""
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data"
    
    print("[MUSIC] Music Genre Classification - Model Evaluation")
    print("=" * 70)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(data_path)
    
    # Load data
    if not evaluator.load_data():
        return
    
    # Load models
    evaluator.load_tabular_models()
    
    # Evaluate models
    evaluator.evaluate_tabular_models()
    
    # Create visualizations
    evaluator.create_performance_comparison()
    evaluator.create_confusion_matrices()
    evaluator.create_roc_curves()
    evaluator.create_class_performance()
    
    # Generate summary
    summary_df = evaluator.generate_summary_report()
    
    print(f"\n[TARGET] Next Steps:")
    print(f"   1. Run: python scripts/compare_approaches.py")
    print(f"   2. Run: streamlit run app.py")
    print(f"   3. Check results in data/processed/")

if __name__ == "__main__":
    main()
