"""
Download MovieLens 100K dataset from GroupLens
"""
import os
import zipfile
import requests
from pathlib import Path

def download_movielens_100k():
    """Download and extract MovieLens 100K dataset"""
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Download URL for MovieLens 100K
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    
    print("Downloading MovieLens 100K dataset...")
    print(f"URL: {url}")
    
    try:
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        zip_path = data_dir / "ml-100k.zip"
        
        # Save the zip file
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded: {zip_path}")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        print("Extraction completed!")
        
        # List extracted files
        extracted_dir = data_dir / "ml-100k"
        if extracted_dir.exists():
            print("\nExtracted files:")
            for file in extracted_dir.iterdir():
                print(f"  - {file.name}")
        
        # Remove zip file to save space
        zip_path.unlink()
        print(f"\nRemoved zip file: {zip_path}")
        
        return str(extracted_dir)
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    dataset_path = download_movielens_100k()
    if dataset_path:
        print(f"\nDataset ready at: {dataset_path}")
    else:
        print("Failed to download dataset")
