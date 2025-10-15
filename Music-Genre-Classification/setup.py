#!/usr/bin/env python3
"""
Setup script for Music Genre Classification project
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = ["data/models", "data/processed", "data/raw", "data/spectrograms", "logs"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory already exists: {directory}")

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required!")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        return True

def check_audio_dependencies():
    """Check audio processing dependencies"""
    print("🎵 Checking audio dependencies...")
    try:
        import librosa
        import soundfile
        print("✅ Audio processing libraries available")
        return True
    except ImportError as e:
        print(f"❌ Missing audio dependencies: {e}")
        return False

def main():
    """Main setup function"""
    print("🎵 Music Genre Classification - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check audio dependencies
    if not check_audio_dependencies():
        print("⚠️ Audio dependencies may need manual installation")
    
    print("\n🎉 Setup completed successfully!")
    print("\n🚀 To run the application:")
    print("   streamlit run app_enhanced.py")
    print("\n📖 For more information, see README.md")

if __name__ == "__main__":
    main()
