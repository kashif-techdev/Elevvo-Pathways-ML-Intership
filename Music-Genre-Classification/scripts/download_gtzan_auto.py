"""
Automatic GTZAN Dataset Download Script

Downloads and organizes the GTZAN dataset using kagglehub.
"""

import kagglehub
import shutil
from pathlib import Path
import os

def download_and_organize_dataset():
    """Download and organize GTZAN dataset"""
    print("GTZAN Dataset Automatic Download")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data"
    raw_path = data_path / "raw"
    
    # Create directories
    raw_path.mkdir(parents=True, exist_ok=True)
    
    print("[DOWNLOAD] Downloading GTZAN dataset from Kaggle...")
    try:
        # Download dataset
        dataset_path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
        print(f"[SUCCESS] Dataset downloaded to: {dataset_path}")
        
        # Find audio files in the downloaded dataset
        audio_files = []
        for ext in ['*.wav', '*.au']:
            audio_files.extend(Path(dataset_path).glob(f"**/{ext}"))
        
        print(f"[INFO] Found {len(audio_files)} audio files")
        
        if len(audio_files) == 0:
            print("[ERROR] No audio files found in downloaded dataset!")
            return False
        
        # Organize files by genre
        genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
                  'jazz', 'metal', 'pop', 'reggae', 'rock']
        
        # Create genre folders
        for genre in genres:
            (raw_path / genre).mkdir(exist_ok=True)
        
        print("[ORGANIZE] Organizing files by genre...")
        organized_count = 0
        
        for file_path in audio_files:
            filename = file_path.stem.lower()
            
            # Determine genre based on filename
            genre = None
            for g in genres:
                if g in filename:
                    genre = g
                    break
            
            if genre:
                target_path = raw_path / genre / file_path.name
                shutil.copy2(file_path, target_path)
                organized_count += 1
            else:
                print(f"[WARNING] Could not determine genre for: {file_path.name}")
        
        print(f"[SUCCESS] Organized {organized_count} files")
        
        # Verify organization
        total_files = 0
        for genre in genres:
            genre_path = raw_path / genre
            files = list(genre_path.glob("*.wav")) + list(genre_path.glob("*.au"))
            file_count = len(files)
            total_files += file_count
            print(f"   {genre}: {file_count} files")
        
        print(f"\n[SUMMARY] Total organized files: {total_files}")
        
        if total_files >= 500:
            print("[SUCCESS] Dataset organization complete!")
            return True
        else:
            print("[WARNING] Dataset appears incomplete.")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        print("[HELP] Make sure you have:")
        print("   1. Internet connection")
        print("   2. Kaggle account (free)")
        print("   3. Kaggle API token configured")
        return False

def main():
    """Main function"""
    success = download_and_organize_dataset()
    
    if success:
        print("\n[READY] Dataset is ready for training!")
        print("[NEXT] Run: python scripts/train_all_models.py")
    else:
        print("\n[ERROR] Dataset download failed.")
        print("[HELP] Try manual download: python scripts/download_gtzan_simple.py")

if __name__ == "__main__":
    main()
