"""
Create sample models for testing the Traffic Sign Recognition web app
This script creates dummy models to test the web interface without full training
"""

import os
import numpy as np
import tensorflow as tf
from models import TrafficSignModelBuilder

def create_sample_models():
    """Create sample models for testing"""
    print("Creating sample models for testing...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize model builder
    builder = TrafficSignModelBuilder(input_shape=(64, 64, 3), num_classes=43)
    
    # Create and save sample models
    models = {
        'custom_cnn': builder.build_custom_cnn(),
        'mobilenet': builder.build_mobilenet_model(),
        'vgg16': builder.build_vgg16_model(),
        'resnet50': builder.build_resnet50_model()
    }
    
    # Compile models
    for name, model in models.items():
        model = builder.compile_model(model)
        
        # Save model
        model_path = f'models/{name}_best.h5'
        model.save(model_path)
        print(f"Created sample model: {model_path}")
        
        # Print model info
        total_params = model.count_params()
        print(f"   Parameters: {total_params:,}")
        print(f"   Model size: {(total_params * 4) / (1024 * 1024):.1f} MB")
        print()
    
    print("Sample models created successfully!")
    print("You can now use the web app to test the interface.")
    print("Note: These are untrained models - for real predictions, train with actual data.")

if __name__ == "__main__":
    create_sample_models()
