"""
Create a sample dataset for testing the Music Genre Classification project
This creates synthetic audio files for testing purposes
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import os

def create_synthetic_audio(duration=30, sr=22050, genre="blues"):
    """Create synthetic audio for testing"""
    
    # Generate different types of synthetic audio based on genre
    if genre == "classical":
        # Classical: slow, harmonic
        t = np.linspace(0, duration, int(sr * duration))
        freq = 440  # A4 note
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        # Add harmonics
        audio += 0.2 * np.sin(2 * np.pi * freq * 2 * t)
        audio += 0.1 * np.sin(2 * np.pi * freq * 3 * t)
        
    elif genre == "rock":
        # Rock: faster, more complex
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.4 * np.sin(2 * np.pi * 220 * t)  # A3
        audio += 0.3 * np.sin(2 * np.pi * 330 * t)  # E4
        # Add some noise
        audio += 0.1 * np.random.normal(0, 1, len(t))
        
    elif genre == "jazz":
        # Jazz: complex harmonies
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.3 * np.sin(2 * np.pi * 261.63 * t)  # C4
        audio += 0.2 * np.sin(2 * np.pi * 329.63 * t)  # E4
        audio += 0.2 * np.sin(2 * np.pi * 392.00 * t)  # G4
        
    else:
        # Default: simple sine wave
        t = np.linspace(0, duration, int(sr * duration))
        freq = 440 + hash(genre) % 200  # Different frequency per genre
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
    
    # Add some variation over time
    envelope = np.exp(-t / duration * 2)  # Decay envelope
    audio *= envelope
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio

def create_sample_dataset():
    """Create a sample dataset for testing"""
    print("Creating Sample Dataset for Testing")
    print("=" * 40)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data"
    raw_path = data_path / "raw"
    raw_path.mkdir(parents=True, exist_ok=True)
    
    # Genres
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    # Create genre folders
    for genre in genres:
        genre_path = raw_path / genre
        genre_path.mkdir(exist_ok=True)
        print(f"[FOLDER] Created: {genre}")
    
    # Create sample files
    print("\n[CREATING] Sample audio files...")
    files_created = 0
    
    for genre in genres:
        genre_path = raw_path / genre
        
        # Create 5 sample files per genre (instead of 100)
        for i in range(5):
            filename = f"{genre}.{i:03d}.wav"
            filepath = genre_path / filename
            
            try:
                # Generate synthetic audio
                audio = create_synthetic_audio(duration=30, genre=genre)
                
                # Save as WAV file
                sf.write(str(filepath), audio, 22050)
                files_created += 1
                
                if files_created % 10 == 0:
                    print(f"[PROGRESS] Created {files_created} files...")
                    
            except Exception as e:
                print(f"[ERROR] Failed to create {filename}: {e}")
    
    print(f"\n[SUCCESS] Created {files_created} sample audio files")
    print(f"[LOCATION] Files saved to: {raw_path}")
    
    print("\n[NOTE] These are synthetic audio files for testing purposes.")
    print("For real genre classification, download the actual GTZAN dataset:")
    print("https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification")
    
    print("\n[NEXT] You can now run:")
    print("1. python scripts/feature_extraction.py")
    print("2. python scripts/train_all_models.py")
    print("3. streamlit run app.py")

if __name__ == "__main__":
    create_sample_dataset()

