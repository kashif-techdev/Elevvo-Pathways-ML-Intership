"""
Professional Launcher for Movie Recommendation System
Enhanced startup with system checks and beautiful UI
"""
import subprocess
import sys
import os
from pathlib import Path

def print_banner():
    """Print professional startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║  🎬 MOVIE RECOMMENDATION SYSTEM 🎬                                           ║
    ║                                                                              ║
    ║  Professional AI-Powered Movie Recommendations                               ║
    ║  Built with Streamlit, Scikit-learn, and Python                           ║
    ║                                                                              ║
    ║  Features:                                                                   ║
    ║  • User-Based Collaborative Filtering                                        ║
    ║  • Item-Based Collaborative Filtering                                       ║
    ║  • SVD Matrix Factorization                                                 ║
    ║  • Professional UI with Animations                                          ║
    ║  • Real-time Data Visualizations                                            ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_system():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Check required files
    required_files = ["app.py", "data_preprocessing.py", "collaborative_filtering.py"]
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Missing file: {file}")
            return False
    
    print("✅ All required files found")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                             "streamlit", "pandas", "numpy", "scikit-learn", 
                             "matplotlib", "seaborn", "requests"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def download_dataset():
    """Download dataset if not present"""
    data_path = Path("data/ml-100k")
    if not data_path.exists():
        print("📥 Downloading MovieLens 100K dataset...")
        try:
            from download_dataset import download_movielens_100k
            download_movielens_100k()
            print("✅ Dataset downloaded successfully")
        except Exception as e:
            print(f"❌ Failed to download dataset: {e}")
            return False
    else:
        print("✅ Dataset already exists")
    return True

def launch_app():
    """Launch the Streamlit application"""
    print("🚀 Launching Movie Recommendation System...")
    print("🌐 The app will open in your default browser")
    print("📍 URL: http://localhost:8501")
    print("\n" + "="*80)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to launch Streamlit app")
        return False
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return True
    
    return True

def main():
    """Main launcher function"""
    print_banner()
    
    # System checks
    if not check_system():
        print("❌ System check failed. Please fix the issues above.")
        return
    
    # Install dependencies
    try:
        import streamlit
        print("✅ Streamlit already installed")
    except ImportError:
        if not install_dependencies():
            print("❌ Failed to install dependencies")
            return
    
    # Download dataset
    if not download_dataset():
        print("❌ Failed to download dataset")
        return
    
    print("\n🎉 All systems ready!")
    print("🚀 Starting Movie Recommendation System...")
    
    # Launch the application
    launch_app()

if __name__ == "__main__":
    main()
