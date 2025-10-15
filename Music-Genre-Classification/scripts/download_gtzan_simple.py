"""
Simple GTZAN Dataset Download Script
Downloads a sample of the GTZAN dataset for testing
"""

import os
import requests
import zipfile
from pathlib import Path
import urllib.request
from tqdm import tqdm

def download_sample_dataset():
    """Download a sample dataset for testing"""
    print("Music Genre Classification - Sample Dataset Download")
    print("=" * 60)
    
    # Create data directory
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data"
    raw_path = data_path / "raw"
    raw_path.mkdir(parents=True, exist_ok=True)
    
    # Create genre folders
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    for genre in genres:
        (raw_path / genre).mkdir(exist_ok=True)
        print(f"[FOLDER] Created: {genre}")
    
    print("\n[INFO] For the full GTZAN dataset, you need to download it manually:")
    print("1. Go to: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification")
    print("2. Download the dataset")
    print("3. Extract all .wav files to the 'data/raw' folder")
    print("4. Run: python scripts/organize_dataset.py")
    
    print("\n[ALTERNATIVE] You can also use a smaller sample dataset for testing:")
    print("1. Download from: https://archive.ics.uci.edu/ml/datasets/GTZAN+Music+Genre+Collection")
    print("2. Or use the sample files provided in the project")
    
    # Create a simple test file structure
    print("\n[SETUP] Creating test structure...")
    
    # Create some dummy files for testing (you can replace these with real audio files)
    for genre in genres:
        genre_path = raw_path / genre
        # Create a placeholder file
        placeholder_file = genre_path / f"{genre}.txt"
        with open(placeholder_file, 'w') as f:
            f.write(f"Placeholder for {genre} audio files\n")
            f.write("Replace this with actual .wav files from the GTZAN dataset\n")
    
    print(f"\n[SUCCESS] Dataset structure created at: {raw_path}")
    print("\n[NEXT] To continue:")
    print("1. Add real .wav files to each genre folder")
    print("2. Run: python scripts/organize_dataset.py")
    print("3. Run: python scripts/feature_extraction.py")

if __name__ == "__main__":
    download_sample_dataset()