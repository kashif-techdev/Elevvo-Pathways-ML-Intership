"""
🎵 Music Genre Classification - Dataset Download Script

Downloads and organizes the GTZAN Music Genre Dataset.
Creates proper folder structure for 10 music genres.
"""

import os
import requests
import zipfile
import shutil
from pathlib import Path
import urllib.request
from tqdm import tqdm

def download_file(url, filename):
    """Download file with progress bar"""
    print(f"[DOWNLOAD] Downloading {filename}...")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) / total_size)
            print(f"\r[PROGRESS] {percent:.1f}%", end="", flush=True)
    
    urllib.request.urlretrieve(url, filename, reporthook=progress_hook)
    print(f"\n[SUCCESS] Downloaded {filename}")

def create_genre_folders(base_path):
    """Create folders for each music genre"""
    genres = [
        'blues', 'classical', 'country', 'disco', 'hiphop',
        'jazz', 'metal', 'pop', 'reggae', 'rock'
    ]
    
    for genre in genres:
        genre_path = base_path / genre
        genre_path.mkdir(parents=True, exist_ok=True)
        print(f"[FOLDER] Created folder: {genre}")
    
    return genres

def organize_dataset(source_path, target_path, genres):
    """Organize downloaded files into genre folders"""
    print("[ORGANIZE] Organizing dataset...")
    
    # Create target directories
    for genre in genres:
        (target_path / genre).mkdir(parents=True, exist_ok=True)
    
    # Find all audio files in source directory
    audio_files = []
    for ext in ['*.wav', '*.au']:
        audio_files.extend(source_path.glob(ext))
    
    print(f"[INFO] Found {len(audio_files)} audio files")
    
    # Organize files by genre
    for file_path in tqdm(audio_files, desc="Organizing files"):
        filename = file_path.stem.lower()
        
        # Determine genre based on filename
        genre = None
        for g in genres:
            if g in filename:
                genre = g
                break
        
        if genre:
            target_file = target_path / genre / file_path.name
            shutil.copy2(file_path, target_file)
        else:
            print(f"[WARNING] Could not determine genre for: {file_path.name}")
    
    print("[SUCCESS] Dataset organization complete!")

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
    
    # GTZAN Dataset URLs (alternative sources)
    dataset_urls = [
        "https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification",
        "https://github.com/mdeff/fma",
        "https://archive.ics.uci.edu/ml/datasets/GTZAN+Music+Genre+Collection"
    ]
    
    print("[INFO] GTZAN Dataset Information:")
    print("   • 1000 audio files (10 genres × 100 songs)")
    print("   • 30 seconds per track")
    print("   • WAV format")
    print("   • Genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock")
    print()
    
    print("[WARNING] Manual Download Required:")
    print("   The GTZAN dataset needs to be downloaded manually from:")
    print("   1. Kaggle: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification")
    print("   2. UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/GTZAN+Music+Genre+Collection")
    print()
    print("[FOLDER] Please download and extract the dataset to:")
    print(f"   {raw_path}")
    print()
    
    # Create genre folders
    genres = create_genre_folders(raw_path)
    
    print("[NEXT] Next Steps:")
    print("   1. Download GTZAN dataset from one of the sources above")
    print("   2. Extract all audio files to the 'data/raw' folder")
    print("   3. Run: python scripts/organize_dataset.py")
    print("   4. Run: python scripts/feature_extraction.py")
    
    # Create organize script
    organize_script = """
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def organize_gtzan_dataset():
    \"\"\"Organize GTZAN dataset into genre folders\"\"\"
    project_root = Path(__file__).parent.parent
    raw_path = project_root / "data" / "raw"
    
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    # Create genre folders
    for genre in genres:
        (raw_path / genre).mkdir(exist_ok=True)
    
    # Find all audio files
    audio_files = list(raw_path.glob("*.wav")) + list(raw_path.glob("*.au"))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Organize by genre
    for file_path in tqdm(audio_files, desc="Organizing"):
        filename = file_path.stem.lower()
        
        # Determine genre
        genre = None
        for g in genres:
            if g in filename:
                genre = g
                break
        
        if genre:
            target = raw_path / genre / file_path.name
            if not target.exists():
                shutil.move(str(file_path), str(target))
        else:
            print(f"Could not determine genre for: {file_path.name}")

if __name__ == "__main__":
    organize_gtzan_dataset()
    print("[SUCCESS] Dataset organization complete!")
"""
    
    organize_script_path = project_root / "scripts" / "organize_dataset.py"
    with open(organize_script_path, 'w') as f:
        f.write(organize_script)
    
    print(f"[SUCCESS] Created organize script: {organize_script_path}")

if __name__ == "__main__":
    main()
