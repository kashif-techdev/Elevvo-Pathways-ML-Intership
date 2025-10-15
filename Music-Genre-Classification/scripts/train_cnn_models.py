"""
[MUSIC] Music Genre Classification - CNN Models Training Script

Trains CNN models for spectrogram-based approach.
Implements custom CNN and transfer learning models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CNNModelTrainer:
    """Train and evaluate CNN models for spectrogram classification"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.history = {}
        
    def load_data(self):
        """Load spectrogram data"""
        processed_path = self.data_path / "processed"
        
        # Load image split data
        img_split_path = processed_path / "image_train_test_split.pkl"
        if not img_split_path.exists():
            print("[ERROR] Image train/test split not found!")
            print("Please run: python scripts/train_test_split.py")
            return None
        
        print("[CHART] Loading spectrogram data...")
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
    
    def create_data_generators(self, batch_size=32, img_size=(128, 128)):
        """Create data generators for training and validation"""
        print(f"\n[GEAR] Creating data generators (batch_size={batch_size}, img_size={img_size})...")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
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
    
    def create_custom_cnn(self, input_shape=(128, 128, 3), num_classes=10):
        """Create custom CNN model"""
        print("\n[BUILDING] Creating custom CNN model...")
        
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense Layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"   • Total parameters: {model.count_params():,}")
        
        return model
    
    def create_transfer_learning_model(self, base_model_name='MobileNetV2', 
                                     input_shape=(128, 128, 3), num_classes=10):
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
    
    def train_model(self, model, model_name, train_gen, val_gen, epochs=50):
        """Train a single model"""
        print(f"\n🚀 Training {model_name}...")
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            callbacks.ModelCheckpoint(
                filepath=self.data_path / "models" / f"{model_name.lower().replace(' ', '_')}_best.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Store results
        self.models[model_name] = model
        self.history[model_name] = history.history
        
        # Evaluate model
        val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
        
        self.results[model_name] = {
            'model': model,
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'history': history.history
        }
        
        print(f"   [OK] Validation Accuracy: {val_accuracy:.4f}")
        print(f"   [OK] Validation Loss: {val_loss:.4f}")
        
        return model, history
    
    def plot_training_history(self):
        """Plot training history for all models"""
        print("\n[CHART] Creating training history plots...")
        
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
        plots_path = self.data_path / "processed" / "cnn_training_history.png"
        plt.savefig(plots_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Saved training history to: {plots_path}")
        plt.show()
    
    def evaluate_models(self, val_gen):
        """Evaluate all trained models"""
        print("\n[CHART] CNN Model Evaluation Results:")
        print("=" * 60)
        print(f"{'Model':<25} {'Val Accuracy':<15} {'Val Loss':<15}")
        print("-" * 60)
        
        for name, result in self.results.items():
            print(f"{name:<25} {result['val_accuracy']:<15.4f} {result['val_loss']:<15.4f}")
        
        # Find best model
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['val_accuracy'])
        best_accuracy = self.results[best_model_name]['val_accuracy']
        
        print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        return best_model_name
    
    def save_models(self):
        """Save trained models"""
        print("\n[SAVED] Saving CNN models...")
        
        models_path = self.data_path / "models"
        models_path.mkdir(parents=True, exist_ok=True)
        
        for name, result in self.results.items():
            model_path = models_path / f"{name.lower().replace(' ', '_')}.h5"
            result['model'].save(model_path)
            print(f"   [SAVED] Saved {name} to: {model_path}")
        
        # Save results summary
        results_summary = {}
        for name, result in self.results.items():
            results_summary[name] = {
                'val_accuracy': result['val_accuracy'],
                'val_loss': result['val_loss']
            }
        
        results_path = models_path / "cnn_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"[SAVED] Saved results to: {results_path}")

def main():
    """Main function"""
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data"
    
    print("[MUSIC] Music Genre Classification - CNN Models Training")
    print("=" * 70)
    
    # Initialize trainer
    trainer = CNNModelTrainer(data_path)
    
    # Load data
    if not trainer.load_data():
        return
    
    # Create data generators
    train_gen, val_gen = trainer.create_data_generators()
    
    # Create and train models
    models_to_train = [
        ('Custom CNN', trainer.create_custom_cnn()),
        ('MobileNetV2 Transfer', trainer.create_transfer_learning_model('MobileNetV2')),
        ('ResNet50 Transfer', trainer.create_transfer_learning_model('ResNet50'))
    ]
    
    for model_name, model in models_to_train:
        trainer.train_model(model, model_name, train_gen, val_gen, epochs=30)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate models
    best_model = trainer.evaluate_models(val_gen)
    
    # Save models
    trainer.save_models()
    
    print(f"\n[TARGET] Next Steps:")
    print(f"   1. Run: python scripts/transfer_learning.py")
    print(f"   2. Run: python scripts/evaluate_models.py")
    print(f"   3. Run: streamlit run app.py")

if __name__ == "__main__":
    main()
