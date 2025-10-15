"""
Traffic Sign Recognition - Training and Evaluation Module
Handles model training, evaluation, and result visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import pandas as pd
import time
import os

class TrafficSignTrainer:
    """Handles training and evaluation of traffic sign recognition models"""
    
    def __init__(self, class_names=None):
        self.class_names = class_names
        self.history = None
        self.model = None
    
    def train_model(self, model, X_train, y_train, X_val, y_val, 
                   epochs=50, batch_size=32, use_augmentation=True, 
                   augmentation_generator=None, callbacks=None):
        """
        Train the model with optional data augmentation
        
        Args:
            model: Keras model to train
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
            use_augmentation: Whether to use data augmentation
            augmentation_generator: Data generator for augmentation
            callbacks: List of training callbacks
            
        Returns:
            Training history
        """
        print(f"Training model for {epochs} epochs...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        start_time = time.time()
        
        if use_augmentation and augmentation_generator is not None:
            # Use data augmentation
            steps_per_epoch = len(X_train) // batch_size
            self.history = model.fit(
                augmentation_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Train without augmentation
            self.history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        self.model = model
        return self.history
    
    def evaluate_model(self, X_test, y_test, model=None):
        """
        Evaluate the model on test data
        
        Args:
            X_test, y_test: Test data
            model: Model to evaluate (uses self.model if None)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model is None:
            model = self.model
        
        print("Evaluating model on test data...")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_true_classes, y_pred_classes)
        
        # Top-3 accuracy
        top3_accuracy = tf.keras.metrics.top_k_categorical_accuracy(y_test, y_pred, k=3)
        top3_accuracy = np.mean(top3_accuracy)
        
        # Top-5 accuracy
        top5_accuracy = tf.keras.metrics.top_k_categorical_accuracy(y_test, y_pred, k=5)
        top5_accuracy = np.mean(top5_accuracy)
        
        results = {
            'test_accuracy': test_accuracy,
            'top3_accuracy': top3_accuracy,
            'top5_accuracy': top5_accuracy,
            'y_true': y_true_classes,
            'y_pred': y_pred_classes,
            'y_pred_proba': y_pred
        }
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Top-3 Accuracy: {top3_accuracy:.4f}")
        print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
        
        return results
    
    def plot_training_history(self, history=None, save_path=None):
        """
        Plot training history (loss and accuracy curves)
        
        Args:
            history: Training history (uses self.history if None)
            save_path: Path to save the plot
        """
        if history is None:
            history = self.history
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(history.history['loss'], label='Training Loss')
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, 
                            save_path=None, figsize=(15, 12)):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_path: Path to save the plot
            figsize: Figure size
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        
        if class_names:
            # Create DataFrame for better labeling
            cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', 
                       cbar_kws={'shrink': 0.8})
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_classification_report(self, y_true, y_pred, class_names=None, 
                                 save_path=None):
        """
        Plot classification report as heatmap
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_path: Path to save the plot
        """
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Extract metrics for each class
        metrics = ['precision', 'recall', 'f1-score']
        classes = [str(i) for i in range(len(set(y_true)))]
        
        if class_names:
            classes = [class_names[int(i)] for i in classes]
        
        # Create DataFrame
        report_df = pd.DataFrame(index=classes, columns=metrics)
        
        for i, class_name in enumerate(classes):
            if str(i) in report:
                report_df.loc[class_name, 'precision'] = report[str(i)]['precision']
                report_df.loc[class_name, 'recall'] = report[str(i)]['recall']
                report_df.loc[class_name, 'f1-score'] = report[str(i)]['f1-score']
        
        # Convert to numeric
        report_df = report_df.astype(float)
        
        # Plot heatmap
        plt.figure(figsize=(10, len(classes) * 0.5))
        sns.heatmap(report_df, annot=True, fmt='.3f', cmap='YlOrRd',
                   cbar_kws={'shrink': 0.8})
        plt.title('Classification Report')
        plt.xlabel('Metrics')
        plt.ylabel('Classes')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_predictions(self, X_test, y_true, y_pred, y_pred_proba, 
                            class_names=None, num_samples=16, save_path=None):
        """
        Visualize sample predictions
        
        Args:
            X_test: Test images
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            class_names: List of class names
            num_samples: Number of samples to display
            save_path: Path to save the plot
        """
        # Get random samples
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            # Display image
            axes[i].imshow(X_test[idx])
            
            # Get prediction info
            true_label = y_true[idx]
            pred_label = y_pred[idx]
            confidence = y_pred_proba[idx][pred_label]
            
            # Format labels
            true_text = class_names[true_label] if class_names else f'True: {true_label}'
            pred_text = class_names[pred_label] if class_names else f'Pred: {pred_label}'
            
            # Color based on correctness
            color = 'green' if true_label == pred_label else 'red'
            
            axes[i].set_title(f'{true_text}\n{pred_text}\nConf: {confidence:.3f}', 
                            color=color, fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_maps(self, model, X_sample, layer_name, num_filters=16, 
                         save_path=None):
        """
        Visualize feature maps from a specific layer
        
        Args:
            model: Trained model
            X_sample: Sample input image
            layer_name: Name of the layer to visualize
            num_filters: Number of filters to display
            save_path: Path to save the plot
        """
        # Create a model that outputs the feature maps
        feature_extractor = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output
        )
        
        # Get feature maps
        feature_maps = feature_extractor.predict(X_sample.reshape(1, *X_sample.shape))
        
        # Plot feature maps
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(min(num_filters, len(axes))):
            axes[i].imshow(feature_maps[0, :, :, i], cmap='viridis')
            axes[i].set_title(f'Filter {i}')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compare_models_performance(self, results_dict, save_path=None):
        """
        Compare performance of different models
        
        Args:
            results_dict: Dictionary with model names and their results
            save_path: Path to save the plot
        """
        models = list(results_dict.keys())
        accuracies = [results_dict[model]['test_accuracy'] for model in models]
        top3_accuracies = [results_dict[model]['top3_accuracy'] for model in models]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        axes[0].bar(models, accuracies, color='skyblue', alpha=0.7)
        axes[0].set_title('Model Accuracy Comparison')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(accuracies):
            axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Top-3 accuracy comparison
        axes[1].bar(models, top3_accuracies, color='lightcoral', alpha=0.7)
        axes[1].set_title('Model Top-3 Accuracy Comparison')
        axes[1].set_ylabel('Top-3 Accuracy')
        axes[1].set_ylim(0, 1)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(top3_accuracies):
            axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_results(self, results, model_name, save_dir='results'):
        """
        Save model results to files
        
        Args:
            results: Dictionary with results
            model_name: Name of the model
            save_dir: Directory to save results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        if self.model:
            self.model.save(f'{save_dir}/{model_name}.h5')
        
        # Save results as CSV
        results_df = pd.DataFrame([results])
        results_df.to_csv(f'{save_dir}/{model_name}_results.csv', index=False)
        
        # Save training history
        if self.history:
            history_df = pd.DataFrame(self.history.history)
            history_df.to_csv(f'{save_dir}/{model_name}_history.csv', index=False)
        
        print(f"Results saved to {save_dir}/")
    
    def load_model(self, model_path):
        """
        Load a saved model
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        self.model = tf.keras.models.load_model(model_path)
        return self.model
