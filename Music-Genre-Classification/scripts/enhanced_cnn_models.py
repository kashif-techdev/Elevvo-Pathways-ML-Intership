"""
🎵 Music Genre Classification - Enhanced CNN Models

Advanced CNN implementation with multiple architectures and training strategies.
Includes custom CNNs, transfer learning, and ensemble methods.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16, EfficientNetB0, DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class EnhancedCNNModelTrainer:
    """Enhanced CNN trainer with multiple architectures and advanced techniques"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.models_path = data_path / "models"
        self.processed_path = data_path / "processed"
        self.models = {}
        self.results = {}
        self.history = {}
        
    def load_data(self):
        """Load spectrogram data and splits"""
        print("Loading spectrogram data...")
        
        # Load image split data
        img_split_path = self.processed_path / "image_train_test_split.pkl"
        if not img_split_path.exists():
            print("ERROR: Image train/test split not found!")
            print("Please run: python scripts/train_test_split.py")
            return False
        
        img_data = joblib.load(img_split_path)
        
        self.X_train_paths = img_data['X_train']
        self.X_test_paths = img_data['X_test']
        self.y_train = img_data['y_train']
        self.y_test = img_data['y_test']
        
        # Load label encoder
        encoder_path = self.processed_path / "label_encoder.pkl"
        self.label_encoder = joblib.load(encoder_path)
        
        print(f"   SUCCESS: Train images: {len(self.X_train_paths)}")
        print(f"   SUCCESS: Test images: {len(self.X_test_paths)}")
        print(f"   SUCCESS: Classes: {len(self.label_encoder.classes_)}")
        
        return True
    
    def create_advanced_data_generators(self, batch_size=32, img_size=(224, 224)):
        """Create advanced data generators with sophisticated augmentation"""
        print(f"\n🔄 Creating advanced data generators (batch_size={batch_size}, img_size={img_size})...")
        
        # Advanced augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.3,
            brightness_range=[0.7, 1.3],
            shear_range=0.2,
            channel_shift_range=0.2,
            fill_mode='nearest'
        )
        
        # Validation generator (no augmentation)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.data_path / "spectrograms" / "mel",
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            self.data_path / "spectrograms" / "mel",
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        print(f"   ✅ Training samples: {train_generator.samples}")
        print(f"   ✅ Validation samples: {val_generator.samples}")
        print(f"   ✅ Classes: {train_generator.num_classes}")
        
        return train_generator, val_generator
    
    def create_advanced_cnn(self, input_shape=(224, 224, 3), num_classes=10):
        """Create advanced custom CNN with modern architecture"""
        print("\n🏗️ Creating advanced custom CNN...")
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fifth Convolutional Block
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            # Dense Layers
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile with advanced optimizer
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        print(f"   ✅ Total parameters: {model.count_params():,}")
        
        return model
    
    def create_residual_cnn(self, input_shape=(224, 224, 3), num_classes=10):
        """Create CNN with residual connections"""
        print("\n🏗️ Creating Residual CNN...")
        
        def residual_block(x, filters, kernel_size=3):
            """Residual block with skip connection"""
            shortcut = x
            
            x = layers.Conv2D(filters, kernel_size, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            x = layers.Conv2D(filters, kernel_size, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Skip connection
            if shortcut.shape[-1] != filters:
                shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            
            x = layers.Add()([x, shortcut])
            x = layers.Activation('relu')(x)
            
            return x
        
        # Input
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(64, 7, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Residual blocks
        x = residual_block(x, 64)
        x = layers.MaxPooling2D(2)(x)
        
        x = residual_block(x, 128)
        x = layers.MaxPooling2D(2)(x)
        
        x = residual_block(x, 256)
        x = layers.MaxPooling2D(2)(x)
        
        x = residual_block(x, 512)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, x)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        print(f"   ✅ Total parameters: {model.count_params():,}")
        
        return model
    
    def create_attention_cnn(self, input_shape=(224, 224, 3), num_classes=10):
        """Create CNN with attention mechanism"""
        print("\n🏗️ Creating Attention CNN...")
        
        def attention_block(x, filters):
            """Spatial attention block"""
            # Global average pooling
            gap = layers.GlobalAveragePooling2D()(x)
            gap = layers.Dense(filters // 4, activation='relu')(gap)
            gap = layers.Dense(filters, activation='sigmoid')(gap)
            
            # Reshape for broadcasting
            gap = layers.Reshape((1, 1, filters))(gap)
            
            # Apply attention
            x = layers.Multiply()([x, gap])
            return x
        
        # Input
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Attention blocks
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = attention_block(x, 128)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = attention_block(x, 256)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = attention_block(x, 512)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, x)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        print(f"   ✅ Total parameters: {model.count_params():,}")
        
        return model
    
    def create_transfer_learning_model(self, base_model_name='EfficientNetB0', 
                                     input_shape=(224, 224, 3), num_classes=10):
        """Create advanced transfer learning model"""
        print(f"\n🏗️ Creating {base_model_name} transfer learning model...")
        
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
        elif base_model_name == 'DenseNet121':
            base_model = DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        else:
            raise ValueError(f"Unknown base model: {base_model_name}")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom head with advanced architecture
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
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
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        print(f"   ✅ Total parameters: {model.count_params():,}")
        print(f"   ✅ Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
        
        return model
    
    def train_model_with_advanced_callbacks(self, model, model_name, train_gen, val_gen, epochs=100):
        """Train model with advanced callbacks and techniques"""
        print(f"\n🚀 Training {model_name} with advanced techniques...")
        
        # Advanced callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=self.models_path / f"{model_name.lower().replace(' ', '_')}_best.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            callbacks.CSVLogger(
                filename=self.models_path / f"{model_name.lower().replace(' ', '_')}_training.log"
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
        val_loss, val_accuracy, val_top3_accuracy = model.evaluate(val_gen, verbose=0)
        
        self.results[model_name] = {
            'model': model,
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'val_top3_accuracy': val_top3_accuracy,
            'history': history.history
        }
        
        print(f"   ✅ Validation Accuracy: {val_accuracy:.4f}")
        print(f"   ✅ Validation Loss: {val_loss:.4f}")
        print(f"   ✅ Top-3 Accuracy: {val_top3_accuracy:.4f}")
        
        return model, history
    
    def create_ensemble_model(self, models_dict, val_gen):
        """Create ensemble model from multiple trained models"""
        print("\n🏗️ Creating ensemble model...")
        
        # Get predictions from all models
        ensemble_predictions = []
        model_names = list(models_dict.keys())
        
        for model_name, model in models_dict.items():
            print(f"   Getting predictions from {model_name}...")
            predictions = model.predict(val_gen, verbose=0)
            ensemble_predictions.append(predictions)
        
        # Average predictions
        ensemble_predictions = np.mean(ensemble_predictions, axis=0)
        
        # Get true labels
        true_labels = val_gen.classes
        
        # Calculate ensemble accuracy
        predicted_labels = np.argmax(ensemble_predictions, axis=1)
        ensemble_accuracy = accuracy_score(true_labels, predicted_labels)
        
        print(f"   ✅ Ensemble Accuracy: {ensemble_accuracy:.4f}")
        
        return ensemble_predictions, ensemble_accuracy
    
    def plot_advanced_training_history(self):
        """Plot advanced training history with multiple metrics"""
        print("\n📊 Creating advanced training history plots...")
        
        n_models = len(self.history)
        fig, axes = plt.subplots(3, n_models, figsize=(6*n_models, 15))
        
        if n_models == 1:
            axes = axes.reshape(3, 1)
        
        for i, (name, history) in enumerate(self.history.items()):
            # Plot accuracy
            axes[0, i].plot(history['accuracy'], label='Training', linewidth=2)
            axes[0, i].plot(history['val_accuracy'], label='Validation', linewidth=2)
            axes[0, i].set_title(f'{name} - Accuracy', fontsize=14, fontweight='bold')
            axes[0, i].set_xlabel('Epoch')
            axes[0, i].set_ylabel('Accuracy')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Plot loss
            axes[1, i].plot(history['loss'], label='Training', linewidth=2)
            axes[1, i].plot(history['val_loss'], label='Validation', linewidth=2)
            axes[1, i].set_title(f'{name} - Loss', fontsize=14, fontweight='bold')
            axes[1, i].set_xlabel('Epoch')
            axes[1, i].set_ylabel('Loss')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
            
            # Plot top-3 accuracy if available
            if 'val_top_3_accuracy' in history:
                axes[2, i].plot(history['top_3_accuracy'], label='Training', linewidth=2)
                axes[2, i].plot(history['val_top_3_accuracy'], label='Validation', linewidth=2)
                axes[2, i].set_title(f'{name} - Top-3 Accuracy', fontsize=14, fontweight='bold')
                axes[2, i].set_xlabel('Epoch')
                axes[2, i].set_ylabel('Top-3 Accuracy')
                axes[2, i].legend()
                axes[2, i].grid(True, alpha=0.3)
            else:
                axes[2, i].text(0.5, 0.5, 'Top-3 Accuracy\nNot Available', 
                              ha='center', va='center', transform=axes[2, i].transAxes)
                axes[2, i].set_title(f'{name} - Top-3 Accuracy', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plots_path = self.processed_path / "enhanced_cnn_training_history.png"
        plt.savefig(plots_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved training history to: {plots_path}")
        plt.show()
    
    def create_model_comparison_plot(self):
        """Create comprehensive model comparison plot"""
        print("\n📊 Creating model comparison plot...")
        
        # Prepare data
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['val_accuracy'] for name in model_names]
        losses = [self.results[name]['val_loss'] for name in model_names]
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        bars1 = ax1.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax1.set_title('CNN Models - Validation Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Validation Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, accuracy in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{accuracy:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Loss comparison
        bars2 = ax2.bar(model_names, losses, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax2.set_title('CNN Models - Validation Loss Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Validation Loss')
        
        # Add value labels on bars
        for bar, loss in zip(bars2, losses):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{loss:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        plots_path = self.processed_path / "enhanced_cnn_comparison.png"
        plt.savefig(plots_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved comparison plot to: {plots_path}")
        plt.show()
    
    def save_enhanced_models(self):
        """Save all trained models with metadata"""
        print("\n💾 Saving enhanced CNN models...")
        
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        for name, result in self.results.items():
            model_path = self.models_path / f"{name.lower().replace(' ', '_')}_enhanced.h5"
            result['model'].save(model_path)
            print(f"   ✅ Saved {name} to: {model_path}")
        
        # Save comprehensive results
        results_summary = {}
        for name, result in self.results.items():
            results_summary[name] = {
                'val_accuracy': result['val_accuracy'],
                'val_loss': result['val_loss'],
                'val_top3_accuracy': result.get('val_top3_accuracy', 0),
                'total_parameters': result['model'].count_params()
            }
        
        results_path = self.models_path / "enhanced_cnn_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"✅ Saved results to: {results_path}")

def main():
    """Main function to train enhanced CNN models"""
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data"
    
    print("Music Genre Classification - Enhanced CNN Models")
    print("=" * 70)
    
    # Initialize trainer
    trainer = EnhancedCNNModelTrainer(data_path)
    
    # Load data
    if not trainer.load_data():
        return
    
    # Create data generators
    train_gen, val_gen = trainer.create_advanced_data_generators()
    
    # Define models to train
    models_to_train = [
        ('Advanced CNN', trainer.create_advanced_cnn()),
        ('Residual CNN', trainer.create_residual_cnn()),
        ('Attention CNN', trainer.create_attention_cnn()),
        ('EfficientNetB0 Transfer', trainer.create_transfer_learning_model('EfficientNetB0')),
        ('DenseNet121 Transfer', trainer.create_transfer_learning_model('DenseNet121'))
    ]
    
    # Train all models
    for model_name, model in models_to_train:
        trainer.train_model_with_advanced_callbacks(model, model_name, train_gen, val_gen, epochs=50)
    
    # Create ensemble
    ensemble_predictions, ensemble_accuracy = trainer.create_ensemble_model(trainer.models, val_gen)
    print(f"\n🏆 Ensemble Model Accuracy: {ensemble_accuracy:.4f}")
    
    # Create visualizations
    trainer.plot_advanced_training_history()
    trainer.create_model_comparison_plot()
    
    # Save models
    trainer.save_enhanced_models()
    
    print(f"\n🎯 Enhanced CNN Training Complete!")
    print(f"   • Models trained: {len(trainer.results)}")
    print(f"   • Best model: {max(trainer.results.keys(), key=lambda x: trainer.results[x]['val_accuracy'])}")
    print(f"   • Ensemble accuracy: {ensemble_accuracy:.4f}")

if __name__ == "__main__":
    main()
