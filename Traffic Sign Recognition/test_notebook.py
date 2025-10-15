"""
Test script to verify the Traffic Sign Recognition notebook components work
"""

import numpy as np
import os
import sys

# Add current directory to path
sys.path.append('.')

def test_imports():
    """Test if all imports work"""
    print("Testing imports...")
    
    try:
        from data_preprocessing import TrafficSignDataProcessor
        from models import TrafficSignModelBuilder
        from training_evaluation import TrafficSignTrainer
        print("Custom modules imported successfully")
        return True
    except Exception as e:
        print(f"Import error: {e}")
        return False

def test_data_processor():
    """Test data processor"""
    print("\nTesting data processor...")
    
    try:
        from data_preprocessing import TrafficSignDataProcessor
        processor = TrafficSignDataProcessor(img_size=(64, 64))
        
        # Test preprocessing
        sample_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processed = processor.preprocess_image(sample_image)
        
        print(f"Data processor working - processed shape: {processed.shape}")
        return True
    except Exception as e:
        print(f"Data processor error: {e}")
        return False

def test_model_builder():
    """Test model builder"""
    print("\nTesting model builder...")
    
    try:
        from models import TrafficSignModelBuilder
        builder = TrafficSignModelBuilder(input_shape=(64, 64, 3), num_classes=43)
        
        # Test custom CNN
        model = builder.build_custom_cnn()
        model = builder.compile_model(model)
        
        print(f"Model builder working - parameters: {model.count_params():,}")
        return True
    except Exception as e:
        print(f"Model builder error: {e}")
        return False

def test_sample_data():
    """Test sample data creation"""
    print("\nTesting sample data creation...")
    
    try:
        # Create sample data
        X_train = np.random.random((100, 64, 64, 3)).astype(np.float32)
        y_train = np.random.randint(0, 43, 100)
        
        print(f"Sample data created - shape: {X_train.shape}, labels: {len(np.unique(y_train))}")
        return True
    except Exception as e:
        print(f"Sample data error: {e}")
        return False

def main():
    """Run all tests"""
    print("Traffic Sign Recognition - Component Testing")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_processor,
        test_model_builder,
        test_sample_data
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! The notebook should work correctly.")
        print("\nNext steps:")
        print("1. Run: jupyter notebook Traffic_Sign_Recognition.ipynb")
        print("2. Run: streamlit run streamlit_app.py")
    else:
        print("Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()
