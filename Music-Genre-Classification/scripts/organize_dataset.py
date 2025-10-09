
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def organize_gtzan_dataset():
    """Organize GTZAN dataset into genre folders"""
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
