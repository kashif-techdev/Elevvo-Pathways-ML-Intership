"""
Dataset Verification Script

Verifies that the GTZAN dataset is properly downloaded and organized.
"""

import os
from pathlib import Path

def verify_dataset():
    """Verify GTZAN dataset structure and files"""
    print("GTZAN Dataset Verification")
    print("=" * 40)
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "raw"
    
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    total_files = 0
    missing_genres = []
    
    print(f"[CHECK] Verifying dataset in: {data_path}")
    print()
    
    for genre in genres:
        genre_path = data_path / genre
        
        if not genre_path.exists():
            print(f"[ERROR] Genre folder not found: {genre}")
            missing_genres.append(genre)
            continue
        
        # Count audio files
        audio_files = list(genre_path.glob("*.wav")) + list(genre_path.glob("*.au"))
        file_count = len(audio_files)
        total_files += file_count
        
        if file_count == 0:
            print(f"[ERROR] No audio files found in: {genre}")
            missing_genres.append(genre)
        elif file_count < 50:
            print(f"[WARNING] {genre}: {file_count} files (expected ~100)")
        else:
            print(f"[SUCCESS] {genre}: {file_count} files")
    
    print()
    print(f"[SUMMARY] Total audio files: {total_files}")
    
    if total_files == 0:
        print("[ERROR] No audio files found!")
        print("[ACTION] Please download the GTZAN dataset first.")
        print("   Run: python scripts/download_gtzan_simple.py")
        return False
    elif total_files < 500:
        print("[WARNING] Dataset appears incomplete.")
        print("[ACTION] Please ensure all genres have audio files.")
        return False
    else:
        print("[SUCCESS] Dataset verification passed!")
        print("[NEXT] You can now run: python scripts/train_all_models.py")
        return True

def main():
    """Main function"""
    success = verify_dataset()
    
    if not success:
        print("\n[HELP] For download instructions, run:")
        print("   python scripts/download_gtzan_simple.py")
        exit(1)
    else:
        print("\n[READY] Dataset is ready for training!")

if __name__ == "__main__":
    main()
