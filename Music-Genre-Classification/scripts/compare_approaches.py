"""
[MUSIC] Music Genre Classification - Approach Comparison Script

Compares feature-based vs image-based approaches.
Generates comprehensive comparison analysis and visualizations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ApproachComparator:
    """Compare feature-based vs image-based approaches"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.processed_path = data_path / "processed"
        self.models_path = data_path / "models"
        self.comparison_results = {}
        
    def load_results(self):
        """Load results from both approaches"""
        print("[CHART] Loading results from both approaches...")
        
        # Load tabular results
        tabular_results_path = self.models_path / "tabular_results.json"
        if tabular_results_path.exists():
            with open(tabular_results_path, 'r') as f:
                self.tabular_results = json.load(f)
            print(f"   [OK] Loaded {len(self.tabular_results)} tabular models")
        else:
            self.tabular_results = {}
            print("   [WARNING]  Tabular results not found")
        
        # Load CNN results
        cnn_results_path = self.models_path / "cnn_results.json"
        if cnn_results_path.exists():
            with open(cnn_results_path, 'r') as f:
                self.cnn_results = json.load(f)
            print(f"   [OK] Loaded {len(self.cnn_results)} CNN models")
        else:
            self.cnn_results = {}
            print("   [WARNING]  CNN results not found")
        
        # Load dataset statistics
        stats_path = self.processed_path / "feature_stats.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                self.dataset_stats = json.load(f)
        else:
            self.dataset_stats = {}
    
    def create_approach_comparison(self):
        """Create comprehensive approach comparison"""
        print("\n[CHART] Creating approach comparison...")
        
        # Prepare data for comparison
        comparison_data = []
        
        # Add tabular results
        for model_name, results in self.tabular_results.items():
            comparison_data.append({
                'Model': model_name,
                'Approach': 'Feature-Based',
                'Accuracy': results.get('accuracy', 0),
                'F1-Score': results.get('f1_score', 0),
                'CV_Mean': results.get('cv_mean', 0),
                'CV_Std': results.get('cv_std', 0)
            })
        
        # Add CNN results
        for model_name, results in self.cnn_results.items():
            comparison_data.append({
                'Model': model_name,
                'Approach': 'Image-Based',
                'Accuracy': results.get('val_accuracy', 0),
                'F1-Score': results.get('f1_score', 0),
                'CV_Mean': results.get('cv_mean', 0),
                'CV_Std': results.get('cv_std', 0)
            })
        
        self.comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate approach averages
        approach_stats = self.comparison_df.groupby('Approach').agg({
            'Accuracy': ['mean', 'std'],
            'F1-Score': ['mean', 'std'],
            'CV_Mean': ['mean', 'std']
        }).round(4)
        
        print("\n[CHART] Approach Comparison Summary:")
        print("=" * 60)
        print(approach_stats)
        
        return self.comparison_df, approach_stats
    
    def create_accuracy_comparison(self):
        """Create accuracy comparison visualization"""
        print("\n[CHART] Creating accuracy comparison...")
        
        # Create grouped bar chart
        fig = px.bar(
            self.comparison_df,
            x='Model',
            y='Accuracy',
            color='Approach',
            title='Model Accuracy Comparison',
            labels={'Accuracy': 'Accuracy Score', 'Model': 'Model'},
            color_discrete_sequence=['#1f77b4', '#ff7f0e']
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=600,
            showlegend=True
        )
        
        # Save plot
        plots_path = self.processed_path / "accuracy_comparison.png"
        fig.write_image(str(plots_path))
        print(f"[SAVED] Saved accuracy comparison to: {plots_path}")
        
        return fig
    
    def create_approach_radar_chart(self):
        """Create radar chart comparing approaches"""
        print("\n[CHART] Creating radar chart...")
        
        # Calculate approach averages
        approach_means = self.comparison_df.groupby('Approach').mean()
        
        # Create radar chart
        fig = go.Figure()
        
        metrics = ['Accuracy', 'F1-Score', 'CV_Mean']
        
        for approach in approach_means.index:
            values = [approach_means.loc[approach, metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=approach,
                line_color='blue' if approach == 'Feature-Based' else 'orange'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Approach Performance Radar Chart"
        )
        
        # Save plot
        plots_path = self.processed_path / "approach_radar_chart.png"
        fig.write_image(str(plots_path))
        print(f"[SAVED] Saved radar chart to: {plots_path}")
        
        return fig
    
    def create_model_performance_heatmap(self):
        """Create model performance heatmap"""
        print("\n[CHART] Creating performance heatmap...")
        
        # Pivot data for heatmap
        heatmap_data = self.comparison_df.pivot(
            index='Model', 
            columns='Approach', 
            values='Accuracy'
        )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            heatmap_data, 
            annot=True, 
            fmt='.4f', 
            cmap='YlOrRd',
            ax=ax,
            cbar_kws={'label': 'Accuracy Score'}
        )
        
        ax.set_title('Model Performance Heatmap')
        ax.set_xlabel('Approach')
        ax.set_ylabel('Model')
        
        plt.tight_layout()
        
        # Save plot
        plots_path = self.processed_path / "performance_heatmap.png"
        plt.savefig(plots_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Saved heatmap to: {plots_path}")
        plt.show()
    
    def create_approach_advantages(self):
        """Analyze advantages of each approach"""
        print("\n[CHART] Analyzing approach advantages...")
        
        # Calculate statistics
        feature_based = self.comparison_df[self.comparison_df['Approach'] == 'Feature-Based']
        image_based = self.comparison_df[self.comparison_df['Approach'] == 'Image-Based']
        
        advantages = {
            'Feature-Based': {
                'Best Accuracy': feature_based['Accuracy'].max(),
                'Average Accuracy': feature_based['Accuracy'].mean(),
                'Best Model': feature_based.loc[feature_based['Accuracy'].idxmax(), 'Model'],
                'Stability': 1 - feature_based['CV_Std'].mean(),
                'Training Speed': 'Fast',
                'Interpretability': 'High',
                'Feature Engineering': 'Required'
            },
            'Image-Based': {
                'Best Accuracy': image_based['Accuracy'].max(),
                'Average Accuracy': image_based['Accuracy'].mean(),
                'Best Model': image_based.loc[image_based['Accuracy'].idxmax(), 'Model'],
                'Stability': 1 - image_based['CV_Std'].mean(),
                'Training Speed': 'Slow',
                'Interpretability': 'Low',
                'Feature Engineering': 'Minimal'
            }
        }
        
        # Create comparison table
        comparison_table = pd.DataFrame(advantages).T
        
        print("\n[CHART] Approach Advantages Analysis:")
        print("=" * 80)
        print(comparison_table.to_string())
        
        # Save advantages
        advantages_path = self.processed_path / "approach_advantages.json"
        with open(advantages_path, 'w') as f:
            json.dump(advantages, f, indent=2)
        print(f"[SAVED] Saved advantages to: {advantages_path}")
        
        return advantages
    
    def create_recommendations(self):
        """Generate recommendations based on comparison"""
        print("\n[CHART] Generating recommendations...")
        
        # Find best models
        best_feature_model = self.comparison_df[
            self.comparison_df['Approach'] == 'Feature-Based'
        ].loc[self.comparison_df['Accuracy'].idxmax()]
        
        best_image_model = self.comparison_df[
            self.comparison_df['Approach'] == 'Image-Based'
        ].loc[self.comparison_df['Accuracy'].idxmax()]
        
        # Calculate approach performance
        feature_avg = self.comparison_df[
            self.comparison_df['Approach'] == 'Feature-Based'
        ]['Accuracy'].mean()
        
        image_avg = self.comparison_df[
            self.comparison_df['Approach'] == 'Image-Based'
        ]['Accuracy'].mean()
        
        recommendations = {
            'Best Feature-Based Model': {
                'Model': best_feature_model['Model'],
                'Accuracy': best_feature_model['Accuracy'],
                'Approach': 'Feature-Based'
            },
            'Best Image-Based Model': {
                'Model': best_image_model['Model'],
                'Accuracy': best_image_model['Accuracy'],
                'Approach': 'Image-Based'
            },
            'Overall Best Approach': 'Feature-Based' if feature_avg > image_avg else 'Image-Based',
            'Performance Difference': abs(feature_avg - image_avg),
            'Recommendations': {
                'For Production': 'Feature-Based' if feature_avg > image_avg else 'Image-Based',
                'For Research': 'Both approaches for comprehensive analysis',
                'For Speed': 'Feature-Based',
                'For Accuracy': 'Feature-Based' if feature_avg > image_avg else 'Image-Based'
            }
        }
        
        print("\n[CHART] Recommendations:")
        print("=" * 50)
        print(f"🏆 Best Feature-Based Model: {recommendations['Best Feature-Based Model']['Model']}")
        print(f"🏆 Best Image-Based Model: {recommendations['Best Image-Based Model']['Model']}")
        print(f"[TARGET] Overall Best Approach: {recommendations['Overall Best Approach']}")
        print(f"[CHART] Performance Difference: {recommendations['Performance Difference']:.4f}")
        
        # Save recommendations
        recommendations_path = self.processed_path / "recommendations.json"
        with open(recommendations_path, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"[SAVED] Saved recommendations to: {recommendations_path}")
        
        return recommendations
    
    def create_comprehensive_report(self):
        """Create comprehensive comparison report"""
        print("\n[CHART] Creating comprehensive report...")
        
        # Generate all comparisons
        comparison_df, approach_stats = self.create_approach_comparison()
        
        # Create visualizations
        self.create_accuracy_comparison()
        self.create_approach_radar_chart()
        self.create_model_performance_heatmap()
        
        # Analyze advantages
        advantages = self.create_approach_advantages()
        
        # Generate recommendations
        recommendations = self.create_recommendations()
        
        # Create final report
        report = {
            'comparison_data': comparison_df.to_dict('records'),
            'approach_statistics': approach_stats.to_dict(),
            'advantages': advantages,
            'recommendations': recommendations,
            'dataset_info': self.dataset_stats
        }
        
        # Save comprehensive report
        report_path = self.processed_path / "comprehensive_comparison_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"[SAVED] Saved comprehensive report to: {report_path}")
        
        return report

def main():
    """Main function"""
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data"
    
    print("[MUSIC] Music Genre Classification - Approach Comparison")
    print("=" * 70)
    
    # Initialize comparator
    comparator = ApproachComparator(data_path)
    
    # Load results
    comparator.load_results()
    
    # Create comprehensive comparison
    report = comparator.create_comprehensive_report()
    
    print(f"\n[TARGET] Next Steps:")
    print(f"   1. Run: streamlit run app.py")
    print(f"   2. Check results in data/processed/")
    print(f"   3. Review comprehensive report")

if __name__ == "__main__":
    main()
