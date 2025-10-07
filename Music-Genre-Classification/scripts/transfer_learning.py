"""
🎵 Music Genre Classification - Transfer Learning Script

Implements transfer learning with pre-trained models for enhanced performance.
Uses MobileNetV2, ResNet50, and VGG16 for spectrogram classification.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TransferLearningTrainer:
    """Transfer learning implementation for music genre classification"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.models_path = data_path / "models"
        self.processed_path = data_path / "processed"
        self.transfer_models = {}
        self.results = {}
        self.history = {}
        
    def load_data(self):
        """Load spectrogram data"""
        processed_path = self.processed_path
        
        # Load image split data
        img_split_path = processed_path / "image_train_test_split.pkl"
        if not img_split_path.exists():
            print("❌ Image train/test split not found!")
            print("Please run: python scripts/train_test_split.py")
            return False
        
        print("📊 Loading spectrogram data...")
        import joblib
        img_data = joblib.load(img_split_path)
        
        self.X_train_paths = img_data['X_train']
        self.X_test_paths = img_data['X_test']
        self.y_train = img_data['y_train']
        self.y_test = img_data['y_test']
        
        # Load label encoder
        encoder_path = processed_path / "label_encoder.pkl"
        self.label_encoder = joblib.load(encoder_path)
        
        print(f"   • Train images: {len(self.X_train_paths)}")
        print(f"   • Test images: {len(self.X_test_paths)}")
        print(f"   • Classes: {len(self.label_encoder.classes_)}")
        
        return True
    
    def create_data_generators(self, batch_size=32, img_size=(224, 224)):
        """Create data generators for transfer learning"""
        print(f"\n🔄 Creating data generators (batch_size={batch_size}, img_size={img_size})...")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.data_path / "spectrograms" / "mel",
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        val_generator = val_datagen.flow_from_directory(
            self.data_path / "spectrograms" / "mel",
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        print(f"   • Training samples: {train_generator.samples}")
        print(f"   • Validation samples: {val_generator.samples}")
        print(f"   • Classes: {train_generator.num_classes}")
        
        return train_generator, val_generator
    
    def create_transfer_model(self, base_model_name, input_shape=(224, 224, 3), num_classes=10):
        """Create transfer learning model"""
        print(f"\n🏗️  Creating {base_model_name} transfer learning model...")
        
        # Load pre-trained model
        if base_model_name == 'MobileNetV2':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif base_model_name == 'ResNet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif base_model_name == 'VGG16':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif base_model_name == 'EfficientNetB0':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        else:
            raise ValueError(f"Unknown base model: {base_model_name}")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"   • Total parameters: {model.count_params():,}")
        print(f"   • Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
        
        return model
    
    def fine_tune_model(self, model, model_name, train_gen, val_gen, epochs=50):
        """Fine-tune transfer learning model"""
        print(f"\n🚀 Fine-tuning {model_name}...")
        
        # Phase 1: Train only the head
        print("   Phase 1: Training head layers...")
        
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        # Train head
        history1 = model.fit(
            train_gen,
            epochs=epochs//2,
            validation_data=val_gen,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Phase 2: Unfreeze and fine-tune
        print("   Phase 2: Fine-tuning all layers...")
        
        # Unfreeze base model
        model.layers[0].trainable = True
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Fine-tune
        history2 = model.fit(
            train_gen,
            epochs=epochs//2,
            validation_data=val_gen,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Combine histories
        combined_history = {
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss']
        }
        
        # Store results
        self.transfer_models[model_name] = model
        self.history[model_name] = combined_history
        
        # Evaluate model
        val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
        
        self.results[model_name] = {
            'model': model,
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'history': combined_history
        }
        
        print(f"   ✅ Final Validation Accuracy: {val_accuracy:.4f}")
        print(f"   ✅ Final Validation Loss: {val_loss:.4f}")
        
        return model, combined_history
    
    def create_ensemble_model(self, models_dict):
        """Create ensemble model from multiple transfer learning models"""
        print("\n🏗️  Creating ensemble model...")
        
        # Get predictions from all models
        ensemble_predictions = []
        
        for model_name, model in models_dict.items():
            print(f"   Getting predictions from {model_name}...")
            # This would require test data - placeholder for now
            pass
        
        return ensemble_predictions
    
    def plot_training_history(self):
        """Plot training history for all transfer learning models"""
        print("\n📊 Creating training history plots...")
        
        n_models = len(self.history)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        for i, (name, history) in enumerate(self.history.items()):
            # Plot accuracy
            axes[0, i].plot(history['accuracy'], label='Training')
            axes[0, i].plot(history['val_accuracy'], label='Validation')
            axes[0, i].set_title(f'{name} - Accuracy')
            axes[0, i].set_xlabel('Epoch')
            axes[0, i].set_ylabel('Accuracy')
            axes[0, i].legend()
            axes[0, i].grid(True)
            
            # Plot loss
            axes[1, i].plot(history['loss'], label='Training')
            axes[1, i].plot(history['val_loss'], label='Validation')
            axes[1, i].set_title(f'{name} - Loss')
            axes[1, i].set_xlabel('Epoch')
            axes[1, i].set_ylabel('Loss')
            axes[1, i].legend()
            axes[1, i].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plots_path = self.processed_path / "transfer_learning_history.png"
        plt.savefig(plots_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved training history to: {plots_path}")
        plt.show()
    
    def compare_transfer_models(self):
        """Compare different transfer learning models"""
        print("\n📊 Transfer Learning Model Comparison:")
        print("=" * 60)
        print(f"{'Model':<25} {'Val Accuracy':<15} {'Val Loss':<15}")
        print("-" * 60)
        
        for name, result in self.results.items():
            print(f"{name:<25} {result['val_accuracy']:<15.4f} {result['val_loss']:<15.4f}")
        
        # Find best model
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['val_accuracy'])
        best_accuracy = self.results[best_model_name]['val_accuracy']
        
        print(f"\n🏆 Best Transfer Learning Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        return best_model_name
    
    def create_model_comparison_plot(self):
        """Create comparison plot for all models"""
        print("\n📊 Creating model comparison plot...")
        
        # Prepare data
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['val_accuracy'] for name in model_names]
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        # Add value labels on bars
        for bar, accuracy in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{accuracy:.4f}', ha='center', va='bottom')
        
        ax.set_title('Transfer Learning Models Performance Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel('Validation Accuracy')
        ax.set_ylim(0, 1)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        plots_path = self.processed_path / "transfer_models_comparison.png"
        plt.savefig(plots_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved comparison plot to: {plots_path}")
        plt.show()
    
    def save_models(self):
        """Save transfer learning models"""
        print("\n💾 Saving transfer learning models...")
        
        models_path = self.data_path / "models"
        models_path.mkdir(parents=True, exist_ok=True)
        
        for name, result in self.results.items():
            model_path = models_path / f"{name.lower().replace(' ', '_')}_transfer.h5"
            result['model'].save(model_path)
            print(f"   💾 Saved {name} to: {model_path}")
        
        # Save results summary
        results_summary = {}
        for name, result in self.results.items():
            results_summary[name] = {
                'val_accuracy': result['val_accuracy'],
                'val_loss': result['val_loss']
            }
        
        results_path = models_path / "transfer_learning_results.json"
        import json
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"💾 Saved results to: {results_path}")

def main():
    """Main function"""
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data"
    
    print("🎵 Music Genre Classification - Transfer Learning")
    print("=" * 70)
    
    # Initialize trainer
    trainer = TransferLearningTrainer(data_path)
    
    # Load data
    if not trainer.load_data():
        return
    
    # Create data generators
    train_gen, val_gen = trainer.create_data_generators()
    
    # Define transfer learning models
    transfer_models = [
        ('MobileNetV2 Transfer', 'MobileNetV2'),
        ('ResNet50 Transfer', 'ResNet50'),
        ('VGG16 Transfer', 'VGG16'),
        ('EfficientNetB0 Transfer', 'EfficientNetB0')
    ]
    
    # Train transfer learning models
    for model_name, base_model_name in transfer_models:
        print(f"\n🎯 Training {model_name}...")
        
        # Create model
        model = trainer.create_transfer_model(base_model_name)
        
        # Fine-tune model
        trainer.fine_tune_model(model, model_name, train_gen, val_gen, epochs=30)
    
    # Create visualizations
    trainer.plot_training_history()
    trainer.create_model_comparison_plot()
    
    # Compare models
    best_model = trainer.compare_transfer_models()
    
    # Save models
    trainer.save_models()
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Run: python scripts/compare_approaches.py")
    print(f"   2. Run: streamlit run app.py")
    print(f"   3. Check results in data/models/")

if __name__ == "__main__":
    main()
