"""
Traffic Sign Recognition - Model Building Module
Contains custom CNN and transfer learning models
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

class TrafficSignModelBuilder:
    """Builds and configures different types of models for traffic sign recognition"""
    
    def __init__(self, input_shape=(64, 64, 3), num_classes=43):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build_custom_cnn(self, dropout_rate=0.5):
        """
        Build a custom CNN from scratch
        
        Args:
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(dropout_rate),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_mobilenet_model(self, freeze_base=True, dropout_rate=0.5):
        """
        Build a model using MobileNetV2 as base
        
        Args:
            freeze_base: Whether to freeze the base model weights
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled Keras model
        """
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model if specified
        if freeze_base:
            base_model.trainable = False
        
        # Build the model
        model = Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_vgg16_model(self, freeze_base=True, dropout_rate=0.5):
        """
        Build a model using VGG16 as base
        
        Args:
            freeze_base: Whether to freeze the base model weights
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled Keras model
        """
        # Load pre-trained VGG16
        base_model = VGG16(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model if specified
        if freeze_base:
            base_model.trainable = False
        
        # Build the model
        model = Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_resnet50_model(self, freeze_base=True, dropout_rate=0.5):
        """
        Build a model using ResNet50 as base
        
        Args:
            freeze_base: Whether to freeze the base model weights
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled Keras model
        """
        # Load pre-trained ResNet50
        base_model = ResNet50(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model if specified
        if freeze_base:
            base_model.trainable = False
        
        # Build the model
        model = Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, model, learning_rate=0.001):
        """
        Compile the model with optimizer, loss, and metrics
        
        Args:
            model: Keras model to compile
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled model
        """
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_callbacks(self, model_name='model', patience=10, monitor='val_accuracy'):
        """
        Get training callbacks
        
        Args:
            model_name: Name for saving the model
            patience: Patience for early stopping
            monitor: Metric to monitor
            
        Returns:
            List of callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f'{model_name}_best.h5',
                monitor=monitor,
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def unfreeze_and_fine_tune(self, model, learning_rate=1e-5):
        """
        Unfreeze the base model for fine-tuning
        
        Args:
            model: Model with frozen base
            learning_rate: Lower learning rate for fine-tuning
            
        Returns:
            Model ready for fine-tuning
        """
        # Unfreeze the base model
        for layer in model.layers[0].layers:
            layer.trainable = True
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_model_summary(self, model):
        """
        Print model summary and return parameter count
        
        Args:
            model: Keras model
            
        Returns:
            Total number of parameters
        """
        model.summary()
        
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        
        return total_params, trainable_params, non_trainable_params
    
    def compare_models(self, models_dict):
        """
        Compare different models side by side
        
        Args:
            models_dict: Dictionary with model names as keys and models as values
            
        Returns:
            Comparison DataFrame
        """
        import pandas as pd
        
        comparison_data = []
        
        for name, model in models_dict.items():
            total_params, trainable_params, non_trainable_params = self.get_model_summary(model)
            
            comparison_data.append({
                'Model': name,
                'Total Parameters': total_params,
                'Trainable Parameters': trainable_params,
                'Non-trainable Parameters': non_trainable_params,
                'Model Size (MB)': (total_params * 4) / (1024 * 1024)  # Assuming float32
            })
        
        return pd.DataFrame(comparison_data)
