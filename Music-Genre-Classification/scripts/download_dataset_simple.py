"""
Music Genre Classification - Dataset Download Script (Simple Version)

Downloads and organizes the GTZAN Music Genre Dataset.
Creates proper folder structure for 10 music genres.
"""

import os
import shutil
from pathlib import Path

def create_genre_folders(base_path):
    """Create folders for each music genre"""
    genres = [
        'blues', 'classical', 'country', 'disco', 'hiphop',
        'jazz', 'metal', 'pop', 'reggae', 'rock'
    ]
    
    for genre in genres:
        genre_path = base_path / genre
        genre_path.mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {genre}")
    
    return genres

def main():
    """Main function to download and organize dataset"""
    print("Music Genre Classification - Dataset Setup")
    print("=" * 50)
    
    # Create paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data"
    raw_path = data_path / "raw"
    
    # Create directories
    raw_path.mkdir(parents=True, exist_ok=True)
    
    print("GTZAN Dataset Information:")
    print("   • 1000 audio files (10 genres × 100 songs)")
    print("   • 30 seconds per track")
    print("   • WAV format")
    print("   • Genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock")
    print()
    
    print("Manual Download Required:")
    print("   The GTZAN dataset needs to be downloaded manually from:")
    print("   1. Kaggle: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification")
    print("   2. UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/GTZAN+Music+Genre+Collection")
    print()
    print("Please download and extract the dataset to:")
    print(f"   {raw_path}")
    print()
    
    # Create genre folders
    genres = create_genre_folders(raw_path)
    
    print("Next Steps:")
    print("   1. Download GTZAN dataset from one of the sources above")
    print("   2. Extract all audio files to the 'data/raw' folder")
    print("   3. Run: python scripts/organize_dataset.py")
    print("   4. Run: python scripts/feature_extraction.py")

if __name__ == "__main__":
    main()
