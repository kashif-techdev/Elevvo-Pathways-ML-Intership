"""
Debug script to identify specific errors in the Traffic Sign Recognition notebook
"""

import sys
import os
import traceback

# Add current directory to path
sys.path.append('.')

def test_notebook_cell_1():
    """Test the first cell (imports)"""
    print("Testing Cell 1: Imports...")
    try:
        # Core libraries
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import cv2
        import os
        import time
        import warnings
        warnings.filterwarnings('ignore')

        # Deep learning libraries
        import tensorflow as tf
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Sklearn for evaluation
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.preprocessing import LabelEncoder

        # Custom modules
        from data_preprocessing import TrafficSignDataProcessor
        from models import TrafficSignModelBuilder
        from training_evaluation import TrafficSignTrainer

        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        print("Libraries imported successfully!")
        print(f"TensorFlow version: {tf.__version__}")
        print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
        return True
    except Exception as e:
        print(f"Error in Cell 1: {e}")
        traceback.print_exc()
        return False

def test_notebook_cell_2():
    """Test the second cell (data loading)"""
    print("\nTesting Cell 2: Data Loading...")
    try:
        from data_preprocessing import TrafficSignDataProcessor
        
        # Initialize data processor
        processor = TrafficSignDataProcessor(img_size=(64, 64))

        # Check if dataset exists locally
        data_dir = 'data'
        train_csv_path = os.path.join(data_dir, 'Train.csv')
        train_images_path = os.path.join(data_dir, 'Train')
        test_csv_path = os.path.join(data_dir, 'Test.csv')
        test_images_path = os.path.join(data_dir, 'Test')

        print("Dataset paths configured:")
        print(f"Training CSV: {train_csv_path}")
        print(f"Training Images: {train_images_path}")
        print(f"Test CSV: {test_csv_path}")
        print(f"Test Images: {test_images_path}")

        # Check if dataset exists
        if os.path.exists(train_csv_path) and os.path.exists(train_images_path):
            print("\nDataset found! Ready to load data.")
            dataset_available = True
        else:
            print("\nDataset not found!")
            print("To use this notebook with real data:")
            print("1. Download GTSRB dataset from: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
            print("2. Extract to 'data/' directory")
            print("3. Run this notebook again")
            print("\nFor now, we'll create sample data for demonstration.")
            dataset_available = False
        
        return True, dataset_available
    except Exception as e:
        print(f"Error in Cell 2: {e}")
        traceback.print_exc()
        return False, False

def test_notebook_cell_3():
    """Test the third cell (sample data creation)"""
    print("\nTesting Cell 3: Sample Data Creation...")
    try:
        import numpy as np
        
        # Create sample data if dataset not available
        print("Creating sample data for demonstration...")
        
        # Create sample training data
        X_train_full = np.random.random((1000, 64, 64, 3)).astype(np.float32)
        y_train_full = np.random.randint(0, 43, 1000)
        
        # Create sample test data
        X_test = np.random.random((200, 64, 64, 3)).astype(np.float32)
        y_test = np.random.randint(0, 43, 200)
        
        print(f"Sample training data created: {X_train_full.shape}")
        print(f"Sample test data created: {X_test.shape}")
        print(f"Number of classes: {len(np.unique(y_train_full))}")
        return True
    except Exception as e:
        print(f"Error in Cell 3: {e}")
        traceback.print_exc()
        return False

def test_notebook_cell_4():
    """Test the fourth cell (data splitting)"""
    print("\nTesting Cell 4: Data Splitting...")
    try:
        from data_preprocessing import TrafficSignDataProcessor
        import numpy as np
        
        # Create sample data
        X_train_full = np.random.random((1000, 64, 64, 3)).astype(np.float32)
        y_train_full = np.random.randint(0, 43, 1000)
        X_test = np.random.random((200, 64, 64, 3)).astype(np.float32)
        y_test = np.random.randint(0, 43, 200)
        
        # Initialize processor
        processor = TrafficSignDataProcessor(img_size=(64, 64))
        
        # Split training data into train, validation, and test sets
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
            X_train_full, y_train_full, 
            test_size=0.2, val_size=0.2, random_state=42
        )

        # Encode labels to categorical format
        y_train_encoded, y_val_encoded, y_test_encoded = processor.encode_labels(
            y_train, y_val, y_test
        )

        print(f"Data split completed:")
        print(f"  Training: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
        print(f"  Image shape: {X_train.shape[1:]}")
        print(f"  Number of classes: {y_train_encoded.shape[1]}")
        return True
    except Exception as e:
        print(f"Error in Cell 4: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all notebook cell tests"""
    print("Traffic Sign Recognition Notebook - Debug Test")
    print("=" * 50)
    
    # Test each cell
    cell1_ok = test_notebook_cell_1()
    cell2_ok, dataset_available = test_notebook_cell_2()
    cell3_ok = test_notebook_cell_3()
    cell4_ok = test_notebook_cell_4()
    
    print(f"\nDebug Results:")
    print(f"Cell 1 (Imports): {'PASS' if cell1_ok else 'FAIL'}")
    print(f"Cell 2 (Data Loading): {'PASS' if cell2_ok else 'FAIL'}")
    print(f"Cell 3 (Sample Data): {'PASS' if cell3_ok else 'FAIL'}")
    print(f"Cell 4 (Data Splitting): {'PASS' if cell4_ok else 'FAIL'}")
    
    if all([cell1_ok, cell2_ok, cell3_ok, cell4_ok]):
        print("\nAll cells working correctly!")
        print("The notebook should run without errors.")
    else:
        print("\nSome cells have errors. Check the output above for details.")

if __name__ == "__main__":
    main()
