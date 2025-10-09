"""
Simple GTZAN Dataset Download Script

This script provides instructions for downloading the GTZAN dataset.
The dataset needs to be downloaded manually due to licensing restrictions.
"""

import os
from pathlib import Path

def main():
    """Main function to guide dataset download"""
    print("GTZAN Dataset Download Guide")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "raw"
    
    print("\n[INFO] GTZAN Dataset Information:")
    print("   • 1000 audio files (10 genres × 100 songs)")
    print("   • 30 seconds per track")
    print("   • WAV format")
    print("   • Genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock")
    
    print("\n[DOWNLOAD] Please download the dataset from one of these sources:")
    print("   1. Kaggle: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification")
    print("   2. UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/GTZAN+Music+Genre+Collection")
    print("   3. Alternative: https://github.com/mdeff/fma")
    
    print(f"\n[FOLDER] Extract all audio files to: {data_path}")
    print("\n[STRUCTURE] The folder structure should look like:")
    print("   data/raw/")
    print("   ├── blues/")
    print("   │   ├── blues.00000.wav")
    print("   │   ├── blues.00001.wav")
    print("   │   └── ...")
    print("   ├── classical/")
    print("   │   ├── classical.00000.wav")
    print("   │   └── ...")
    print("   └── ... (other genres)")
    
    print("\n[VERIFY] After downloading, verify with:")
    print("   python scripts/verify_dataset.py")
    
    print("\n[NEXT] Once downloaded, run:")
    print("   python scripts/train_all_models.py")

if __name__ == "__main__":
    main()
