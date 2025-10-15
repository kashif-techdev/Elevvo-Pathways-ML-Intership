"""
Traffic Sign Recognition - Simple Demo Script
Quick demonstration of the project capabilities without emoji characters
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from data_preprocessing import TrafficSignDataProcessor
from models import TrafficSignModelBuilder
from training_evaluation import TrafficSignTrainer

def demo_data_preprocessing():
    """Demonstrate data preprocessing capabilities"""
    print("Data Preprocessing Demo")
    print("=" * 30)
    
    # Initialize processor
    processor = TrafficSignDataProcessor(img_size=(64, 64))
    
    # Get class names
    class_names = processor.get_class_names()
    print(f"Number of classes: {len(class_names)}")
    print(f"Sample classes: {list(class_names.values())[:5]}")
    
    # Create sample image
    sample_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Preprocess image
    processed_image = processor.preprocess_image(sample_image)
    
    print(f"Original shape: {sample_image.shape}")
    print(f"Processed shape: {processed_image.shape}")
    print(f"Pixel range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
    
    return processor

def demo_model_building():
    """Demonstrate model building capabilities"""
    print("\nModel Building Demo")
    print("=" * 30)
    
    # Initialize model builder
    builder = TrafficSignModelBuilder(input_shape=(64, 64, 3), num_classes=43)
    
    # Build different models
    models = {
        'Custom CNN': builder.build_custom_cnn(),
        'MobileNetV2': builder.build_mobilenet_model(),
        'VGG16': builder.build_vgg16_model(),
        'ResNet50': builder.build_resnet50_model()
    }
    
    # Compile models
    for name, model in models.items():
        model = builder.compile_model(model)
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        
        print(f"{name}:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: {(total_params * 4) / (1024 * 1024):.1f} MB")
    
    return models

def demo_training_simulation():
    """Demonstrate training capabilities with simulated data"""
    print("\nTraining Simulation Demo")
    print("=" * 30)
    
    # Create simulated data
    X_train = np.random.random((1000, 64, 64, 3)).astype(np.float32)
    y_train = np.random.randint(0, 43, 1000)
    y_train_encoded = tf.keras.utils.to_categorical(y_train, 43)
    
    X_val = np.random.random((200, 64, 64, 3)).astype(np.float32)
    y_val = np.random.randint(0, 43, 200)
    y_val_encoded = tf.keras.utils.to_categorical(y_val, 43)
    
    # Build a simple model for demo
    builder = TrafficSignModelBuilder()
    model = builder.build_custom_cnn()
    model = builder.compile_model(model)
    
    # Initialize trainer
    trainer = TrafficSignTrainer()
    
    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    print(f"Model architecture: {model.name}")
    
    # Simulate training (just a few epochs for demo)
    print("Simulating training...")
    
    # This would normally take much longer
    print("Training simulation completed!")
    print("In real training, you would see:")
    print("   - Loss decreasing over epochs")
    print("   - Accuracy increasing over epochs")
    print("   - Validation metrics for monitoring")
    
    return trainer

def demo_evaluation():
    """Demonstrate evaluation capabilities"""
    print("\nEvaluation Demo")
    print("=" * 30)
    
    # Create simulated test data
    X_test = np.random.random((500, 64, 64, 3)).astype(np.float32)
    y_test = np.random.randint(0, 43, 500)
    y_test_encoded = tf.keras.utils.to_categorical(y_test, 43)
    
    # Create a simple model for demo
    builder = TrafficSignModelBuilder()
    model = builder.build_custom_cnn()
    model = builder.compile_model(model)
    
    # Simulate predictions
    predictions = model.predict(X_test[:10])  # Just first 10 samples
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = y_test[:10]
    
    # Calculate accuracy
    accuracy = np.mean(predicted_classes == true_classes)
    
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Sample predictions: {predicted_classes[:5]}")
    print(f"True labels: {true_classes[:5]}")
    
    # Show confidence scores
    confidence_scores = np.max(predictions, axis=1)
    print(f"Average confidence: {np.mean(confidence_scores):.3f}")
    
    return accuracy

def main():
    """Main demo function"""
    print("Traffic Sign Recognition - Project Demo")
    print("=" * 50)
    print("This demo showcases the main capabilities of the project.")
    print("For full functionality, use the complete dataset and training pipeline.")
    print()
    
    try:
        # Demo data preprocessing
        processor = demo_data_preprocessing()
        
        # Demo model building
        models = demo_model_building()
        
        # Demo training simulation
        trainer = demo_training_simulation()
        
        # Demo evaluation
        accuracy = demo_evaluation()
        
        print("\nDemo completed successfully!")
        print("To run the full project:")
        print("   1. Download the GTSRB dataset")
        print("   2. Run: jupyter notebook Traffic_Sign_Recognition.ipynb")
        print("   3. Or run: streamlit run streamlit_app.py")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
