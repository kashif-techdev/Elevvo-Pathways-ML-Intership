"""
Music Genre Classification - Spectrogram Generation Script

Creates mel spectrograms from audio files for the image-based approach.
Converts audio to visual representations for CNN classification.
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SpectrogramGenerator:
    """Generate mel spectrograms from audio files"""
    
    def __init__(self, sample_rate=22050, duration=30, n_mels=128, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def create_mel_spectrogram(self, y, sr):
        """Create mel spectrogram from audio signal"""
        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale
        S_db = librosa.power_to_db(S, ref=np.max)
        
        return S_db
    
    def create_chromagram(self, y, sr):
        """Create chromagram from audio signal"""
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        return chroma
    
    def create_mfcc_spectrogram(self, y, sr, n_mfcc=20):
        """Create MFCC spectrogram"""
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfcc
    
    def save_spectrogram(self, file_path, output_path, spectrogram_type='mel'):
        """Save spectrogram as image"""
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # Create spectrogram based on type
            if spectrogram_type == 'mel':
                S = self.create_mel_spectrogram(y, sr)
            elif spectrogram_type == 'chroma':
                S = self.create_chromagram(y, sr)
            elif spectrogram_type == 'mfcc':
                S = self.create_mfcc_spectrogram(y, sr)
            else:
                raise ValueError(f"Unknown spectrogram type: {spectrogram_type}")
            
            # Create figure
            plt.figure(figsize=(3, 3), dpi=100)
            plt.axis('off')
            
            # Display spectrogram
            if spectrogram_type == 'mel':
                librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='mel')
            elif spectrogram_type == 'chroma':
                librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='chroma')
            elif spectrogram_type == 'mfcc':
                librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='mel')
            
            # Save image
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            plt.close()
            return False
    
    def create_spectrogram_dataset(self, data_path, output_path, spectrogram_type='mel'):
        """Create spectrogram dataset from audio files"""
        genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
                  'jazz', 'metal', 'pop', 'reggae', 'rock']
        
        total_files = 0
        successful_files = 0
        
        print(f"[START] Creating {spectrogram_type} spectrograms...")
        
        for genre in genres:
            genre_input_path = data_path / "raw" / genre
            genre_output_path = output_path / spectrogram_type / genre
            
            if not genre_input_path.exists():
                print(f"[WARNING] Genre folder not found: {genre_input_path}")
                continue
            
            # Create output directory
            genre_output_path.mkdir(parents=True, exist_ok=True)
            
            # Get audio files
            audio_files = list(genre_input_path.glob("*.wav")) + list(genre_input_path.glob("*.au"))
            total_files += len(audio_files)
            
            print(f"[GENRE] {genre}: {len(audio_files)} files")
            
            for file_path in tqdm(audio_files, desc=f"Processing {genre}"):
                # Create output filename
                output_filename = file_path.stem + '.png'
                output_file_path = genre_output_path / output_filename
                
                # Skip if already exists
                if output_file_path.exists():
                    successful_files += 1
                    continue
                
                # Create spectrogram
                success = self.save_spectrogram(file_path, output_file_path, spectrogram_type)
                if success:
                    successful_files += 1
        
        print(f"\n📈 Spectrogram Generation Complete!")
        print(f"   • Total files: {total_files}")
        print(f"   • Successful: {successful_files}")
        print(f"   • Success rate: {successful_files/total_files*100:.1f}%")
        
        return successful_files, total_files

def create_multiple_spectrogram_types(data_path, output_path):
    """Create different types of spectrograms"""
    generator = SpectrogramGenerator()
    
    spectrogram_types = ['mel', 'chroma', 'mfcc']
    
    for spec_type in spectrogram_types:
        print(f"\n🎨 Creating {spec_type} spectrograms...")
        output_type_path = output_path / spec_type
        output_type_path.mkdir(parents=True, exist_ok=True)
        
        successful, total = generator.create_spectrogram_dataset(
            data_path, output_path, spec_type
        )
        
        print(f"[SUCCESS] {spec_type}: {successful}/{total} files processed")

def create_combined_spectrograms(data_path, output_path):
    """Create combined spectrograms with multiple features"""
    print("\n[INFO] Creating combined spectrograms...")
    
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    for genre in genres:
        genre_input_path = data_path / "raw" / genre
        genre_output_path = output_path / "combined" / genre
        
        if not genre_input_path.exists():
            continue
        
        genre_output_path.mkdir(parents=True, exist_ok=True)
        
        audio_files = list(genre_input_path.glob("*.wav")) + list(genre_input_path.glob("*.au"))
        
        for file_path in tqdm(audio_files, desc=f"Combined {genre}"):
            try:
                # Load audio
                y, sr = librosa.load(file_path, sr=22050, duration=30)
                
                # Create combined spectrogram
                fig, axes = plt.subplots(2, 2, figsize=(6, 6), dpi=100)
                fig.suptitle(f'{genre.title()} - {file_path.stem}', fontsize=10)
                
                # Mel spectrogram
                S_mel = librosa.feature.melspectrogram(y=y, sr=sr)
                S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
                librosa.display.specshow(S_mel_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[0,0])
                axes[0,0].set_title('Mel Spectrogram')
                axes[0,0].axis('off')
                
                # Chromagram
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=axes[0,1])
                axes[0,1].set_title('Chromagram')
                axes[0,1].axis('off')
                
                # MFCC
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=axes[1,0])
                axes[1,0].set_title('MFCC')
                axes[1,0].axis('off')
                
                # Spectral contrast
                contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                librosa.display.specshow(contrast, sr=sr, x_axis='time', ax=axes[1,1])
                axes[1,1].set_title('Spectral Contrast')
                axes[1,1].axis('off')
                
                # Save combined image
                output_file = genre_output_path / f"{file_path.stem}_combined.png"
                plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=100)
                plt.close()
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                plt.close()

def main():
    """Main function"""
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data"
    output_path = data_path / "spectrograms"
    
    # Check if raw data exists
    raw_path = data_path / "raw"
    if not raw_path.exists():
        print("[ERROR] Raw data folder not found!")
        print("Please run: python scripts/download_dataset.py")
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Music Genre Classification - Spectrogram Generation")
    print("=" * 60)
    
    # Create different types of spectrograms
    create_multiple_spectrogram_types(data_path, output_path)
    
    # Create combined spectrograms
    create_combined_spectrograms(data_path, output_path)
    
    print("\n[NEXT] Next Steps:")
    print("   1. Run: python scripts/train_cnn_models.py")
    print("   2. Run: python scripts/transfer_learning.py")
    print("   3. Run: streamlit run app.py")

if __name__ == "__main__":
    main()
