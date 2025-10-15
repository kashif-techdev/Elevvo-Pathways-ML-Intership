"""
Google Colab Setup Script for Traffic Sign Recognition Project
Run this cell first in Google Colab to set up the environment
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    
    packages = [
        'tensorflow>=2.10.0',
        'opencv-python>=4.6.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'scikit-learn>=1.1.0',
        'streamlit>=1.20.0',
        'Pillow>=9.0.0',
        'plotly>=5.10.0'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
    
    print("✅ All packages installed!")

def setup_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        'models',
        'results',
        'data',
        'data/Train',
        'data/Test'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    print("✅ All directories created!")

def download_sample_data():
    """Download sample data if not available"""
    print("📊 Checking for dataset...")
    
    # Check if GTSRB dataset exists
    if not os.path.exists('data/Train.csv'):
        print("⚠️  GTSRB dataset not found!")
        print("📥 Please download the dataset from:")
        print("   https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
        print("📁 Extract the files to the 'data' directory")
        print("   - Train.csv and Train/ folder")
        print("   - Test.csv and Test/ folder")
    else:
        print("✅ Dataset found!")

def check_gpu():
    """Check GPU availability"""
    print("🖥️  Checking GPU availability...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU available: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
        else:
            print("⚠️  No GPU detected - training will be slower")
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")

def main():
    """Main setup function"""
    print("🚀 Setting up Traffic Sign Recognition Project for Google Colab")
    print("=" * 60)
    
    # Install packages
    install_requirements()
    print()
    
    # Create directories
    setup_directories()
    print()
    
    # Check dataset
    download_sample_data()
    print()
    
    # Check GPU
    check_gpu()
    print()
    
    print("🎉 Setup completed successfully!")
    print("📝 Next steps:")
    print("   1. Upload the GTSRB dataset to the 'data' folder")
    print("   2. Run the main notebook: Traffic_Sign_Recognition.ipynb")
    print("   3. Or run the Streamlit app: streamlit run streamlit_app.py")
    print()
    print("🚀 Happy coding!")

if __name__ == "__main__":
    main()
