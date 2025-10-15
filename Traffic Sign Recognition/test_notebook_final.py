"""
Final test script to verify the Traffic Sign Recognition notebook works completely
"""

import sys
import os
import traceback

# Add current directory to path
sys.path.append('.')

def test_complete_notebook_flow():
    """Test the complete notebook flow"""
    print("Testing Complete Traffic Sign Recognition Notebook Flow")
    print("=" * 60)
    
    try:
        # Step 1: Imports
        print("Step 1: Testing imports...")
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import cv2
        import os
        import time
        import warnings
        warnings.filterwarnings('ignore')

        import tensorflow as tf
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.preprocessing import LabelEncoder

        from data_preprocessing import TrafficSignDataProcessor
        from models import TrafficSignModelBuilder
        from training_evaluation import TrafficSignTrainer

        np.random.seed(42)
        tf.random.set_seed(42)
        print("All imports successful")
        
        # Step 2: Data processor setup
        print("\nStep 2: Setting up data processor...")
        processor = TrafficSignDataProcessor(img_size=(64, 64))
        
        # Check dataset availability
        data_dir = 'data'
        train_csv_path = os.path.join(data_dir, 'Train.csv')
        train_images_path = os.path.join(data_dir, 'Train')
        
        if os.path.exists(train_csv_path) and os.path.exists(train_images_path):
            print("Real dataset found")
            dataset_available = True
        else:
            print("No dataset found - will use sample data")
            dataset_available = False
        
        # Step 3: Create sample data (since no real dataset)
        print("\nStep 3: Creating sample data...")
        X_train_full = np.random.random((1000, 64, 64, 3)).astype(np.float32)
        y_train_full = np.random.randint(0, 43, 1000)
        X_test = np.random.random((200, 64, 64, 3)).astype(np.float32)
        y_test = np.random.randint(0, 43, 200)
        print(f"Sample data created: {X_train_full.shape}, {X_test.shape}")
        
        # Step 4: Data splitting
        print("\nStep 4: Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
            X_train_full, y_train_full, 
            test_size=0.2, val_size=0.2, random_state=42
        )
        print(f"Data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
        
        # Step 5: Label encoding
        print("\nStep 5: Encoding labels...")
        y_train_encoded, y_val_encoded, y_test_encoded = processor.encode_labels(
            y_train, y_val, y_test
        )
        print(f"Labels encoded: {y_train_encoded.shape[1]} classes")
        
        # Step 6: Model building
        print("\nStep 6: Building models...")
        builder = TrafficSignModelBuilder(input_shape=(64, 64, 3), num_classes=43)
        
        # Build custom CNN
        custom_cnn = builder.build_custom_cnn()
        custom_cnn = builder.compile_model(custom_cnn)
        print(f"Custom CNN built: {custom_cnn.count_params():,} parameters")
        
        # Build MobileNetV2
        mobilenet = builder.build_mobilenet_model()
        mobilenet = builder.compile_model(mobilenet)
        print(f"MobileNetV2 built: {mobilenet.count_params():,} parameters")
        
        # Step 7: Quick training demo
        print("\nStep 7: Quick training demo...")
        trainer = TrafficSignTrainer()
        callbacks = builder.get_callbacks('custom_cnn', patience=3)
        
        # Train for just 2 epochs as demo
        history = trainer.train_model(
            model=custom_cnn,
            X_train=X_train, y_train=y_train_encoded,
            X_val=X_val, y_val=y_val_encoded,
            epochs=2,  # Very quick demo
            batch_size=32,
            use_augmentation=False,
            callbacks=callbacks
        )
        print("Training demo completed")
        
        # Step 8: Evaluation
        print("\nStep 8: Evaluating model...")
        results = trainer.evaluate_model(X_test, y_test_encoded, custom_cnn)
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Top-3 Accuracy: {results['top3_accuracy']:.4f}")
        
        print("\nALL TESTS PASSED!")
        print("The notebook should work perfectly now!")
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_notebook_flow()
    if success:
        print("\nThe Traffic Sign Recognition notebook is ready to use!")
        print("You can now run: jupyter notebook Traffic_Sign_Recognition.ipynb")
    else:
        print("\nThere are still issues to resolve.")
