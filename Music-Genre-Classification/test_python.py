"""
Test Python installation and dependencies
"""

import sys
import subprocess

def test_python():
    """Test if Python is working"""
    print("Python Version:", sys.version)
    print("Python Executable:", sys.executable)
    return True

def test_dependencies():
    """Test if required packages are available"""
    required_packages = [
        'numpy', 'pandas', 'sklearn', 'tensorflow', 
        'librosa', 'matplotlib', 'seaborn', 'streamlit'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"[OK] {package} - Available")
        except ImportError:
            print(f"[MISSING] {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n[SUCCESS] All dependencies are available!")
        return True

def main():
    """Main test function"""
    print("=" * 50)
    print("Music Genre Classification - Python Test")
    print("=" * 50)
    
    print("\n[TEST 1] Python Installation")
    test_python()
    
    print("\n[TEST 2] Dependencies")
    deps_ok = test_dependencies()
    
    if deps_ok:
        print("\n[SUCCESS] Ready to run the project!")
        print("Next steps:")
        print("1. python scripts/download_dataset.py")
        print("2. python scripts/train_all_models.py")
        print("3. streamlit run app.py")
    else:
        print("\n[WARNING] Please install missing dependencies first")
        print("Run: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
