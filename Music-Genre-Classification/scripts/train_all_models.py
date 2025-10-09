"""
🎵 Music Genre Classification - Complete Training Pipeline

Runs the complete training pipeline for all models.
Executes all training scripts in the correct order.
"""

import subprocess
import sys
from pathlib import Path
import time

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n[START] {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(f"[SUCCESS] {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main training pipeline"""
    project_root = Path(__file__).parent.parent
    scripts_path = project_root / "scripts"
    
    print("Music Genre Classification - Complete Training Pipeline")
    print("=" * 80)
    print("This will run all training scripts in the correct order.")
    print("Make sure you have downloaded the GTZAN dataset first!")
    print()
    
    # Training pipeline
    training_steps = [
        ("download_dataset.py", "Download and organize GTZAN dataset"),
        ("feature_extraction.py", "Extract audio features"),
        ("create_spectrograms.py", "Create mel spectrograms"),
        ("train_test_split.py", "Create train/test splits"),
        ("train_tabular_models.py", "Train tabular models"),
        ("train_cnn_models.py", "Train CNN models"),
        ("transfer_learning.py", "Train transfer learning models"),
        ("evaluate_models.py", "Evaluate all models"),
        ("compare_approaches.py", "Compare approaches")
    ]
    
    successful_steps = 0
    total_steps = len(training_steps)
    
    for script_name, description in training_steps:
        script_path = scripts_path / script_name
        
        if not script_path.exists():
            print(f"[WARNING] Script not found: {script_name}")
            continue
        
        success = run_script(str(script_path), description)
        
        if success:
            successful_steps += 1
        else:
            print(f"\n[ERROR] Pipeline stopped at: {description}")
            print("Please fix the error and run the remaining steps manually.")
            break
        
        # Small delay between steps
        time.sleep(2)
    
    # Summary
    print(f"\n[SUMMARY] Training Pipeline Summary:")
    print(f"   • Completed: {successful_steps}/{total_steps} steps")
    print(f"   • Success rate: {successful_steps/total_steps*100:.1f}%")
    
    if successful_steps == total_steps:
        print("\n[SUCCESS] All training steps completed successfully!")
        print("\n[NEXT] Next Steps:")
        print("   1. Run: streamlit run app.py")
        print("   2. Check results in data/processed/")
        print("   3. Review model performance")
    else:
        print(f"\n[WARNING] Pipeline incomplete. {total_steps - successful_steps} steps remaining.")
        print("Please run the remaining scripts manually.")

if __name__ == "__main__":
    main()
